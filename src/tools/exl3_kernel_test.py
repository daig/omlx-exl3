"""Kernel tests for omlx.exl3 against fp32 references built from the same trellises.

  gemv        TrellisGemvPlan (1-3 members, rows 1/3/8, fp16 and bf16 outputs) vs
              had128(xh_m @ W_inner_m) * svh_m with W_inner from kernels.decode_wt
  rot         vs had128(x * suh_m) / sqrt(128) per member
  norm_rot    norm output vs mx.fast.rms_norm (bitwise), rotations vs the reference
  swiglu_rot  vs had128(silu(gate) * up * suh)

    .venv/bin/python tools/exl3_kernel_test.py
"""
import math

import mlx.core as mx

from omlx.exl3 import kernels as K


def had(v):
    *lead, n = v.shape
    return mx.hadamard_transform(v.reshape(-1, n // 128, 128).astype(mx.float32),
                                 scale=1 / math.sqrt(128)).reshape(*lead, n)


def rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.max(mx.abs(a - b)) / mx.max(mx.abs(b))).item()


def random_trellis(k, n, bits, seed):
    words = k * n * bits // 32
    return mx.random.randint(0, 2 ** 31, (words,), key=mx.random.key(seed)).astype(mx.uint32)


def test_gemv():
    k, cb = 1024, 2
    for bits, ns in [(6, [256]), (6, [256, 384]), (6, [128, 256, 128]), (5, [256, 128]), (7, [384]),
                     (8, [128, 128]), (4, [256]), (3, [128])]:
        trs = [random_trellis(k, n, bits, i) for i, n in enumerate(ns)]
        svh = (mx.random.normal((sum(ns),)) * 0.01).astype(mx.float16)
        ws = [K.decode_wt(t, k, n, bits, cb).astype(mx.float32) for t, n in zip(trs, ns)]
        for rows in (1, 3, 8):
            xh = mx.random.normal((len(ns), rows, k)).astype(mx.float16)
            ref = mx.concatenate([had(xh[m].astype(mx.float32) @ ws[m].T) for m in range(len(ns))], axis=1)
            ref = ref * svh.astype(mx.float32)
            for dt in (mx.float16, mx.bfloat16):
                y = K.TrellisGemvPlan(k, ns, bits, cb, svh, rows, dt)(xh, trs)
                assert y.dtype == dt and y.shape == (rows, sum(ns))
                e = rel(y, ref)
                print(f"gemv {bits}-bit members {ns} rows {rows} {str(dt).split('.')[-1]:8s} rel err {e:.2e}")
                assert e < (2e-3 if dt == mx.float16 else 1e-2), e


def test_rot():
    for rows in (1, 3, 8):
        for k, nm in ((5120, 1), (5120, 2), (5120, 3), (6144, 1)):
            x = (mx.random.normal((rows, k)) * 3).astype(mx.bfloat16)
            w = (1 + 0.1 * mx.random.normal((k,))).astype(mx.bfloat16)
            suh = (mx.random.normal((nm, k)) * 0.5).astype(mx.float16)
            xh = K.rot(x, suh)
            ref = mx.stack([had(x.astype(mx.float32) * suh[m].astype(mx.float32)) for m in range(nm)])
            e1 = rel(xh, ref)
            xn, xh2 = K.norm_rot(x, w, 1e-6, suh)
            rn = mx.fast.rms_norm(x, w, 1e-6)
            ref2 = mx.stack([had(rn.astype(mx.float32) * suh[m].astype(mx.float32)) for m in range(nm)])
            differ = mx.mean((xn != rn).astype(mx.float32)).item()
            e2 = rel(xh2, ref2)
            print(f"rot / norm_rot rows {rows} k {k} nm {nm}: rot {e1:.2e}, norm_rot xh {e2:.2e}, "
                  f"xn differs from rms_norm in {100 * differ:.3f}% of elements")
            assert xh.shape == (nm, rows, k) and xh2.shape == (nm, rows, k)
            assert e1 < 1e-3 and e2 < 1e-3 and differ < 1e-3
        n = 17408
        gu = (mx.random.normal((rows, 2 * n)) * 2).astype(mx.float16)
        suh = (mx.random.normal((n,)) * 0.5).astype(mx.float16)
        g, u = gu[:, :n].astype(mx.float32), gu[:, n:].astype(mx.float32)
        e3 = rel(K.swiglu_rot(gu, suh)[0], had(g * mx.sigmoid(g) * u * suh.astype(mx.float32)))
        print(f"swiglu_rot rows {rows}: {e3:.2e}")
        assert e3 < 1e-3


if __name__ == "__main__":
    mx.random.seed(0)
    test_gemv()
    test_rot()
    print("all ok")
