"""Decode-speed bench for EXL3 checkpoints on oMLX's model code path (mlx-vlm model, no server).

Loads once, prefills --context tokens, then alternates configurations in blocks of decode steps:
  pipelined   tok/s as mlx-lm's generate_step runs: the next step's graph is built while the GPU
              runs the current one (at most two steps in flight)
  sync        per step: CPU graph-build time, and build + encode + GPU + wait
Configurations:
  unfused     input rotations as one kernel per EXL3 linear / sibling group
  fused       rotations fused into the producing RMSNorm and SwiGLU (omlx.exl3.fusion)
  legacy      rotations as plain MLX ops (cast, multiply, hadamard_transform), for comparison
--rows 3 steps 3 tokens at a time (the MTP verify width) instead of 1.
By default oMLX's decode-path Qwen3.5 patches (fused GDN prework, verify-width attention split) are
installed first, as the server does; --stock runs plain mlx-vlm.
--cpu-ops     instead: CPU cost of building (not running) one call of each decode-path op.

    .venv/bin/python tools/exl3_decode_bench.py MODEL_DIR [--configs unfused,fused] [--rows 1]
"""
import argparse
import math
import os
import time

os.environ.setdefault("OMLX_EXL3_FUSE", "1")      # install the fused classes; toggled at runtime below

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from omlx.exl3 import fusion, kernels, linear, load  # noqa: E402

_kernel_rot = linear.rot


def mlx_rot(x, suh):
    """Rotation as plain MLX ops: fp16 cast, fp16 multiply, MLX Hadamard (three launches)."""
    rows, k = x.shape
    nm = suh.shape[0]
    xs = suh[:, None, :] * x.astype(mx.float16)[None]                 # member-major (nm, rows, k)
    return mx.hadamard_transform(xs.reshape(nm * rows, k // 128, 128),
                                 scale=1 / math.sqrt(128)).reshape(nm, rows, k)


def configure(name):
    if name == "native":
        return
    fusion.ENABLED = name == "fused"
    linear.rot = mlx_rot if name == "legacy" else _kernel_rot


def cpu_ops(lm):
    """Python/C++ cost of constructing each op's graph node (nothing is evaluated)."""
    from omlx.exl3.linear import EXL3Linear
    lin = next(m for _, m in lm.named_modules() if isinstance(m, EXL3Linear) and m._group is not None)
    grp = lin._group
    x = mx.random.normal((1, grp.k)).astype(mx.bfloat16)
    w = mx.ones((grp.k,), dtype=mx.bfloat16)
    gu = mx.random.normal((1, 2 * 17408)).astype(mx.float16)
    suh_d = mx.ones((17408,), dtype=mx.float16)
    xh = _kernel_rot(x, grp.suh)
    plan = kernels.TrellisGemvPlan(grp.k, grp.ns, grp.K, grp.cb, grp.svh, 1, mx.bfloat16)
    mx.eval(x, w, gu, suh_d, xh)
    ops = {
        "mx.fast.rms_norm": lambda: mx.fast.rms_norm(x, w, 1e-6),
        "x + x (add)": lambda: x + x,
        "rot (kernel)": lambda: _kernel_rot(x, grp.suh),
        "rot (MLX ops, 3 launches)": lambda: mlx_rot(x, grp.suh),
        "norm_rot": lambda: kernels.norm_rot(x, w, 1e-6, grp.suh),
        "swiglu_rot": lambda: kernels.swiglu_rot(gu, suh_d),
        "gemv plan call": lambda: plan(xh, grp.trs),
    }
    for name, fn in ops.items():
        for _ in range(50):
            fn()
        t0 = time.perf_counter()
        n = 2000
        for _ in range(n):
            fn()
        print(f"  {name:28s} {1e6 * (time.perf_counter() - t0) / n:7.1f} us per call (build only)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--configs", default="unfused,fused")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--steps", type=int, default=64, help="decode steps per block")
    ap.add_argument("--sync-steps", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rows", type=int, default=1, help="tokens per step (1 decode, 3 MTP-verify width)")
    ap.add_argument("--cpu-ops", action="store_true")
    ap.add_argument("--stock", action="store_true", help="skip oMLX's Qwen3.5 decode patches")
    args = ap.parse_args()
    if not args.stock:
        from omlx.patches.qwen35_gdn_prework import apply_qwen35_gdn_prework_patch
        from omlx.patches.qwen35_verify_sdpa_split import apply_qwen35_verify_sdpa_split_patch
        print("oMLX patches:", apply_qwen35_gdn_prework_patch(), apply_qwen35_verify_sdpa_split_patch())

    import json
    from pathlib import Path
    qc = json.loads((Path(args.model) / "config.json").read_text()).get("quantization_config") or {}
    if str(qc.get("quant_method", "")).lower() == "exl3":
        model, _ = load(args.model)
    else:                                    # native MLX checkpoint: only the model path is measured
        from mlx_vlm.utils import load as vlm_load
        model, _ = vlm_load(args.model)
        args.configs = "native"
    lm = model.language_model
    if args.cpu_ops:
        cpu_ops(lm)
        return

    rng = np.random.default_rng(0)
    ids = rng.integers(1000, 150000, size=args.context).tolist()
    cache = lm.make_cache()
    lm._position_ids = lm._rope_deltas = None
    for s in range(0, len(ids), 2048):
        lm(mx.array(ids[s:s + 2048])[None], cache=cache)
        mx.eval([a for c in cache for a in (c.state if isinstance(c.state, (list, tuple)) else [c.state])
                 if isinstance(a, mx.array)])

    def step(y):
        logits = lm(y, cache=cache).logits
        return mx.argmax(logits, axis=-1).astype(mx.int32)          # (1, rows): next inputs

    state = {"y": mx.array([ids[-args.rows:]], dtype=mx.int32)}

    def pipelined(n):
        nxt = step(state["y"])
        mx.async_eval(nxt)
        t0 = time.perf_counter()
        for _ in range(n):
            cur = nxt
            nxt = step(cur)
            mx.async_eval(nxt)
            cur.tolist()
        mx.eval(nxt)
        dt = time.perf_counter() - t0
        state["y"] = nxt
        return n / dt

    def sync(n):
        build = total = 0.0
        y = state["y"]
        for _ in range(n):
            t0 = time.perf_counter()
            nxt = step(y)
            t1 = time.perf_counter()
            mx.eval(nxt)
            t2 = time.perf_counter()
            build += t1 - t0
            total += t2 - t0
            y = nxt
        state["y"] = y
        return 1e3 * build / n, 1e3 * total / n

    configs = args.configs.split(",")
    for c in configs:                    # warm-up: compile every kernel variant once
        configure(c)
        pipelined(8)
    res = {c: {"tps": [], "build": [], "step": []} for c in configs}
    for r in range(args.rounds):
        for c in configs:
            configure(c)
            res[c]["tps"].append(pipelined(args.steps))
            b, t = sync(args.sync_steps)
            res[c]["build"].append(b)
            res[c]["step"].append(t)
        print(f"round {r + 1}: " + " | ".join(f"{c} {res[c]['tps'][-1]:.2f} steps/s" for c in configs), flush=True)
    print(f"\nrows/step {args.rows}, context {args.context}+")
    for c in configs:
        d = res[c]
        print(f"  {c:8s} pipelined {np.median(d['tps']):6.2f} steps/s ({1e3 / np.median(d['tps']):5.2f} ms/step) | "
              f"sync: build {np.median(d['build']):5.2f} ms, step {np.median(d['step']):5.2f} ms")


if __name__ == "__main__":
    main()
