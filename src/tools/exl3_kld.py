#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KL divergence of quantized checkpoints against a BF16 reference, all on MLX.

Metrics follow exllamav3's qbench (eval/qbench): per-token KL(ref || model) over the real
vocabulary, mean / median / p90 / p99, reference-confidence buckets and perplexity on each model's
own logits, plus top-1 agreement. Test sets:
  --trace FILE   a qbench_prompts.py trace (turboderp ships one with the EXL3 quants): variable
                 (context, response) rows, scored on the response positions only
  --wiki FILE    wikitext-2 raw test text (wiki.test.raw, or the raw-v1 zip), chat-framed like
                 qbench's `template: true` classic mode: --wiki-rows rows of --wiki-len tokens

Passes, per row, with every model resident (the reference runs once per row):
  tf      teacher-forced: the row through chunked prefill (the prefill code path)
  floor   the reference again with bf16-rounding multiplicative noise after every decoder layer:
          qbench's noise floor, the KLD that an equivalent-but-different kernel selection shows
  dec<W>  decode path: prefill the context, then feed the first --decode-steps scored positions
          W tokens per step through the real cache (W=1: decode kernels, W=3: MTP verify width).
          Compared with the reference's teacher-forced logits; the reference runs it too.

    .venv/bin/python tools/exl3_kld.py --ref BF16_DIR --model SC6=DIR --model 8bit=DIR \\
        --trace DIR/qbench_prompts_gen.json --floor --decode-steps 128 --out kld_trace
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zipfile
from pathlib import Path

import mlx.core as mx
import numpy as np

BF16_ROUNDING_EPS = 2 ** -9          # qbench's noise-floor scale: one bf16 unit roundoff per layer
CONF_BUCKETS = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.95), (0.95, 1.01)]
BLOCK = 256                          # positions per metric block (bounds fp32 full-vocab temporaries)


# ------ checkpoints ----------------------------------------------------------------------------

def load_lm(path: Path):
    """The language model (mlx-vlm LanguageModel) of a checkpoint, EXL3 or native MLX."""
    cfg = json.loads((path / "config.json").read_text())
    qc = cfg.get("quantization_config") or {}
    if str(qc.get("quant_method", "")).lower() == "exl3":
        from omlx.exl3 import load
        model, _ = load(path)
    else:
        from mlx_vlm.utils import load
        model, _ = load(str(path))
    return model.language_model


def _headers(path: Path) -> dict:
    index = path / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() \
        else ["model.safetensors"]
    out = {}
    for f in files:
        with open(path / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            out.update(json.loads(fh.read(n)))
    out.pop("__metadata__", None)
    return out


def _counted(key: str) -> bool:
    """qbench's size convention: decoder layers + output head; embeddings, vision and MTP excluded."""
    if any(s in key for s in ("visual", "vision", "mtp.")):
        return False
    return ".layers." in key or "lm_head" in key


def weight_stats(path: Path) -> tuple[int, int]:
    """(bytes, elements) of the counted tensors."""
    nbytes = nelem = 0
    for k, v in _headers(path).items():
        if _counted(k):
            a, b = v["data_offsets"]
            nbytes += b - a
            nelem += int(np.prod(v["shape"])) if v["shape"] else 1
    return nbytes, nelem


# ------ test data --------------------------------------------------------------------------------

def trace_rows(path: Path):
    """(ids, a, b) per row: logits positions [a, b) are scored (qbench: the sampled response)."""
    trace = json.loads(path.read_text())
    rows = []
    for r in trace["rows"]:
        seq = r["input_ids"] + r["response_ids"]
        rows.append((seq, len(r["input_ids"]) - 1, len(seq)))
    return rows, trace.get("vocab_size")


def wiki_rows(path: Path, tokenizer_dir: Path, n_rows: int, length: int):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            raw = z.read("wikitext-2-raw/wiki.test.raw").decode("utf-8")
    else:
        raw = path.read_text(encoding="utf-8")
    # qbench joins the HF dataset's non-blank lines with blank lines between them
    text = "\n\n".join(line + "\n" for line in raw.split("\n") if line.strip())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    tokens = tok(text, add_special_tokens=False)["input_ids"]
    # template vars as in turboderp's Qwen3.8 trace (the default effort injects a system prompt)
    prefix = tok.apply_chat_template([{"role": "system", "content": ""},
                                      {"role": "user", "content": "Say something."}],
                                     add_generation_prompt=True, tokenize=True,
                                     enable_thinking=True, reasoning_effort="medium")
    if isinstance(prefix, dict) or hasattr(prefix, "keys"):
        prefix = prefix["input_ids"]
    prefix = list(prefix)
    rows = []
    for s in range(0, len(tokens) - length, length):
        seq = prefix + tokens[s:s + length]
        rows.append((seq, len(prefix), len(seq)))
        if len(rows) >= n_rows:
            break
    return rows, len(tok), tok.decode(prefix)


# ------ forward passes ---------------------------------------------------------------------------

def _reset(lm):
    lm._position_ids = None
    lm._rope_deltas = None


def _cache_arrays(cache):
    out = []
    for c in cache:
        st = c.state
        for x in (st if isinstance(st, (list, tuple)) else [st]):
            if isinstance(x, mx.array):
                out.append(x)
    return out


def _prefill(lm, cache, ids, c0, c1, chunk):
    for s in range(c0, c1, chunk):
        out = lm(mx.array(ids[s:min(c1, s + chunk)])[None], cache=cache)
        mx.eval(_cache_arrays(cache))
    return out


def tf_logits(lm, ids, a, b, V, chunk):
    """Teacher-forced fp32 logits at positions [a, b), via chunked prefill over ids[:b]."""
    cache = lm.make_cache()
    _reset(lm)
    parts = []
    for s in range(0, b, chunk):
        e = min(b, s + chunk)
        logits = lm(mx.array(ids[s:e])[None], cache=cache).logits
        lo = max(a, s)
        if lo < e:
            parts.append(logits[0, lo - s:e - s, :V].astype(mx.float32))
            mx.eval(parts[-1], _cache_arrays(cache))
        else:
            mx.eval(_cache_arrays(cache))
    return mx.concatenate(parts) if len(parts) > 1 else parts[0]


def decode_logits(lm, ids, a, n, width, V, chunk):
    """Prefill ids[:a], then step positions [a, a+n) `width` tokens at a time through the cache."""
    cache = lm.make_cache()
    _reset(lm)
    _prefill(lm, cache, ids, 0, a, chunk)
    parts = []
    for t in range(a, a + n, width):
        w = min(width, a + n - t)
        logits = lm(mx.array(ids[t:t + w])[None], cache=cache).logits
        parts.append(logits[0, :, :V].astype(mx.float32))
        mx.eval(parts[-1], _cache_arrays(cache))
    return mx.concatenate(parts)


class NoiseFloor:
    """Multiplicative N(0, eps) noise on every decoder layer's output (qbench's noise floor)."""

    def __init__(self, eps: float, seed: int):
        self.eps, self.seed = eps, seed

    def __enter__(self):
        from mlx_vlm.models.qwen3_5 import language as q35
        self.cls = q35.Qwen3_5DecoderLayer
        self.orig = orig = self.cls.__call__
        key, eps = [mx.random.key(self.seed)], self.eps

        def noisy(layer, *args, **kwargs):
            h = orig(layer, *args, **kwargs)
            key[0], sub = mx.random.split(key[0])
            n = mx.random.normal(h.shape, dtype=mx.float32, key=sub)
            return (h.astype(mx.float32) * (1.0 + eps * n)).astype(h.dtype)

        self.cls.__call__ = noisy
        return self

    def __exit__(self, *exc):
        self.cls.__call__ = self.orig


# ------ metrics ----------------------------------------------------------------------------------

def _logsoftmax(x):
    x = mx.maximum(x, -200.0)                   # qbench clamps logits the same way
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def ref_stats(logits, targets):
    """Reference log-probs plus per-position confidence, argmax and NLL of the next token."""
    lps, conf, top, nll = [], [], [], []
    for s in range(0, logits.shape[0], BLOCK):
        lp = _logsoftmax(logits[s:s + BLOCK])
        lps.append(lp)
        conf.append(mx.exp(mx.max(lp, axis=-1)))
        top.append(mx.argmax(lp, axis=-1))
        nll.append(_target_nll(lp, targets, s))
        mx.eval(lps[-1], conf[-1], top[-1], nll[-1])
    lp = mx.concatenate(lps)
    return lp, np.array(mx.concatenate(conf)), np.array(mx.concatenate(top)), np.concatenate(
        [np.array(x) for x in nll])


def _target_nll(lp, targets, s):
    """NLL of the next token for rows of lp that have one (targets[i] < 0: no next token)."""
    t = targets[s:s + lp.shape[0]]
    idx = mx.array(np.maximum(t, 0).astype(np.int32))[:, None]
    nll = -mx.take_along_axis(lp, idx, axis=-1)[:, 0]
    return mx.where(mx.array(t >= 0), nll, mx.array(np.nan, dtype=mx.float32))


def compare(ref_lp, ref_top, logits, targets):
    """Per-position KL(ref || model), top-1 agreement and the model's own NLL of the next token."""
    kl, agree, nll = [], [], []
    for s in range(0, logits.shape[0], BLOCK):
        lq = _logsoftmax(logits[s:s + BLOCK])
        lr = ref_lp[s:s + BLOCK]
        k = mx.sum(mx.exp(lr) * (lr - lq), axis=-1)
        top = mx.argmax(lq, axis=-1)
        n = _target_nll(lq, targets, s)
        mx.eval(k, top, n)
        kl.append(np.array(k))
        agree.append(np.array(top) == ref_top[s:s + BLOCK])
        nll.append(np.array(n))
    return np.concatenate(kl), np.concatenate(agree), np.concatenate(nll)


def summarize(kl, agree, nll, conf, row_ids):
    finite = np.isfinite(kl)
    k = kl[finite]
    has_t = np.isfinite(nll)
    res = {
        "tokens": int(kl.size),
        "kld": float(k.mean()),
        "kld_median": float(np.median(k)),
        "kld_p90": float(np.quantile(k, 0.90)),
        "kld_p99": float(np.quantile(k, 0.99)),
        "kld_max": float(k.max()),
        "top1": float(agree.mean()),
        "ppl": float(np.exp(nll[has_t].mean())),
        "kld_buckets": [],
    }
    if not finite.all():
        res["nonfinite"] = int((~finite).sum())
    for lo, hi in CONF_BUCKETS:
        m = finite & (conf >= lo) & (conf < hi)
        res["kld_buckets"].append({"conf": [lo, min(hi, 1.0)], "share": float(m.mean()),
                                   "mean": float(kl[m].mean()) if m.any() else None})
    # 95% interval of the mean from a bootstrap over rows (tokens within a row are correlated)
    rng = np.random.default_rng(0)
    rows = np.unique(row_ids)
    sums = np.array([kl[finite & (row_ids == r)].sum() for r in rows])
    cnts = np.array([(finite & (row_ids == r)).sum() for r in rows])
    pick = rng.integers(0, len(rows), size=(2000, len(rows)))
    boot = sums[pick].sum(1) / cnts[pick].sum(1)
    res["kld_ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return res


# ------ driver -----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--wiki", type=Path)
    ap.add_argument("--wiki-rows", type=int, default=20)
    ap.add_argument("--wiki-len", type=int, default=2048)
    ap.add_argument("--rows", default=None, help="subset a:b of the test rows")
    ap.add_argument("--floor", action="store_true", help="noise-floor pass of the reference")
    ap.add_argument("--decode-steps", type=int, default=0, help="decode-path positions per row (0: off)")
    ap.add_argument("--decode-widths", default="1", help="tokens per decode step, comma separated")
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk (tokens)")
    ap.add_argument("--out", type=Path, required=True, help="output prefix (.json summary, .npz per-token)")
    args = ap.parse_args()

    if bool(args.trace) == bool(args.wiki):
        sys.exit("give exactly one of --trace / --wiki")
    if args.trace:
        rows, V = trace_rows(args.trace)
        dataset = {"kind": "trace", "source": str(args.trace)}
    else:
        rows, V, prefix = wiki_rows(args.wiki, args.ref, args.wiki_rows, args.wiki_len)
        dataset = {"kind": "wiki", "source": str(args.wiki), "rows": args.wiki_rows,
                   "length": args.wiki_len, "chat_prefix": prefix}
    if args.rows:
        a, b = (int(x) if x else None for x in args.rows.split(":"))
        rows = rows[a:b]
    widths = [int(w) for w in args.decode_widths.split(",")] if args.decode_steps else []
    print(f"{dataset['kind']}: {len(rows)} rows, {sum(b - a for _, a, b in rows)} scored positions, vocab {V}",
          flush=True)

    labels = ["bf16"] + [m.split("=", 1)[0] for m in args.model]
    paths = [args.ref] + [Path(m.split("=", 1)[1]).expanduser() for m in args.model]
    ref_bytes, ref_elems = weight_stats(args.ref)
    sizes = {}
    for label, p in zip(labels, paths):
        nb, _ = weight_stats(p)
        sizes[label] = {"gib": nb / 2 ** 30, "bpw": 8 * nb / ref_elems}
    lms = {}
    for label, p in zip(labels, paths):
        t0 = time.time()
        lms[label] = load_lm(p)
        print(f"loaded {label} ({sizes[label]['gib']:.2f} GiB decoder+head, {sizes[label]['bpw']:.2f} bpw) "
              f"in {time.time() - t0:.0f}s", flush=True)
    ref = lms["bf16"]

    passes = ["tf:" + l for l in labels[1:]] + (["floor:bf16"] if args.floor else [])
    for w in widths:
        passes += [f"dec{w}:" + l for l in labels]
    acc = {p: {"kl": [], "agree": [], "nll": [], "conf": [], "row": []} for p in passes}
    acc_ref = {"nll": [], "conf": [], "row": []}
    times = {p: 0.0 for p in passes + ["tf:bf16"]}
    ntoks = {p: 0 for p in passes + ["tf:bf16"]}

    def record(pname, ri, kl, agree, nll, conf):
        d = acc[pname]
        d["kl"].append(kl); d["agree"].append(agree); d["nll"].append(nll); d["conf"].append(conf)
        d["row"].append(np.full(kl.shape, ri, dtype=np.int32))

    def results():
        out = {"dataset": dataset, "vocab": V, "sizes": sizes, "passes": {}, "speed_tok_s": {}}
        if acc_ref["nll"]:
            nll = np.concatenate(acc_ref["nll"])
            out["ref_ppl"] = float(np.exp(nll[np.isfinite(nll)].mean()))
        for p, d in acc.items():
            if d["kl"]:
                out["passes"][p] = summarize(*(np.concatenate(d[k]) for k in ("kl", "agree", "nll", "conf", "row")))
        for p in times:
            if times[p] > 0:
                out["speed_tok_s"][p] = ntoks[p] / times[p]
        return out

    t_start = time.time()
    for ri, (ids, a, b) in enumerate(rows):
        targets = np.array(ids[a + 1:b] + [-1], dtype=np.int64)       # next token per scored position
        t0 = time.time()
        logits = tf_logits(ref, ids, a, b, V, args.chunk)
        times["tf:bf16"] += time.time() - t0
        ntoks["tf:bf16"] += b
        ref_lp, conf, ref_top, ref_nll = ref_stats(logits, targets)
        del logits
        acc_ref["nll"].append(ref_nll); acc_ref["conf"].append(conf)
        acc_ref["row"].append(np.full(conf.shape, ri, dtype=np.int32))

        for p in passes:
            kind, label = p.split(":")
            lm = lms[label]
            t0 = time.time()
            if kind == "tf":
                logits = tf_logits(lm, ids, a, b, V, args.chunk)
                n, ntok = b - a, b
            elif kind == "floor":
                with NoiseFloor(BF16_ROUNDING_EPS, seed=1000 + ri):
                    logits = tf_logits(lm, ids, a, b, V, args.chunk)
                n, ntok = b - a, b
            else:
                n = min(args.decode_steps, b - a)
                logits = decode_logits(lm, ids, a, n, int(kind[3:]), V, args.chunk)
                ntok = n
            mx.eval(logits)
            times[p] += time.time() - t0
            ntoks[p] += ntok
            kl, agree, nll = compare(ref_lp[:n], ref_top[:n], logits, targets[:n])
            record(p, ri, kl, agree, nll, conf[:n])
            del logits
        del ref_lp
        mx.clear_cache()

        res = results()
        args.out.with_suffix(".json").write_text(json.dumps(res, indent=2))
        line = " ".join(f"{p}={res['passes'][p]['kld']:.5f}" for p in passes)
        print(f"row {ri + 1}/{len(rows)} ({b - a} pos, {time.time() - t_start:.0f}s) {line}", flush=True)

    np.savez_compressed(args.out.with_suffix(".npz"),
                        **{f"{p.replace(':', '.')}.{k}": np.concatenate(v) for p, d in acc.items()
                           for k, v in d.items() if v},
                        **{f"ref.{k}": np.concatenate(v) for k, v in acc_ref.items() if v})
    res = results()
    print(f"\nreference (bf16) ppl {res.get('ref_ppl', float('nan')):.4f}")
    print(f"{'pass':14s} {'GiB':>6s} {'bpw':>5s} {'KLD':>9s} {'95% CI':>19s} {'median':>9s} {'p90':>8s} "
          f"{'p99':>8s} {'top-1':>7s} {'ppl':>8s}")
    for p, r in res["passes"].items():
        s = sizes[p.split(":")[1]]
        ci = r["kld_ci95"]
        print(f"{p:14s} {s['gib']:6.2f} {s['bpw']:5.2f} {r['kld']:9.6f} [{ci[0]:.6f}, {ci[1]:.6f}] "
              f"{r['kld_median']:9.6f} {r['kld_p90']:8.5f} {r['kld_p99']:8.4f} {r['top1']:7.4f} {r['ppl']:8.4f}")
    print("speed (tok/s):", {k: round(v, 1) for k, v in res["speed_tok_s"].items()})


if __name__ == "__main__":
    main()
