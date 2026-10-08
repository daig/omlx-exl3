# SPDX-License-Identifier: Apache-2.0
"""EXL3 linear layers for MLX models.

EXL3Linear holds exactly the checkpoint tensors of one ExLlamaV3 linear (trellis, suh/su, svh/sv,
mul1/mcg marker, optional bias) as module parameters, so mlx-vlm/mlx-lm's strict weight loading fills
it like any layer. Forward computes  y = had128( had128(x * suh) @ W_inner ) * svh  (+ bias):

  rows <= MAX_GEMV_ROWS (decode, speculative verify):  the input rotation (one kernel, or none when
      the producer primed it, see fusion.py), then one trellis GEMV kernel that also reduces, applies
      the output rotation and svh, and writes the caller's dtype (kernels.TrellisGemvPlan)
  more rows (prefill):  input rotation, decode W_inner^T to fp16 (kernels.decode_wt, column chunks
      for very wide layers), MLX matmul, output rotation and svh

EXL3SiblingGroup runs same-input siblings (q/k/v, gate/up, in_proj_qkv/z) as one GEMV launch over
their own trellis buffers when they share K and codebook; each member returns its slice, served from
a single-slot cache keyed on the input array's identity.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .kernels import MAX_GEMV_ROWS, TrellisGemvPlan, decode_wt, rot

_HAD_SCALE = 1.0 / math.sqrt(128.0)
PREFILL_COL_CHUNK = 16384          # output columns decoded per chunk on the prefill path


def _had128(v: mx.array) -> mx.array:
    rows, n = v.shape
    return mx.hadamard_transform(v.reshape(rows, n // 128, 128), scale=_HAD_SCALE).reshape(rows, n)


def _unpack_signs(bits: mx.array) -> mx.array:
    """Packed int16 sign bitfield (legacy su/sv) -> +-1 fp16 vector, bit i of word j = element 16j+i."""
    b = bits.view(mx.uint16).astype(mx.uint32)
    expanded = (b[:, None] >> mx.arange(16, dtype=mx.uint32)) & 1
    return (1.0 - 2.0 * expanded.reshape(-1).astype(mx.float16)).astype(mx.float16)


class EXL3Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tensors: dict[str, tuple[tuple[int, ...], mx.Dtype]]):
        """tensors: checkpoint tensor name (trellis, suh, svh, su, sv, mul1, mcg, bias) -> (shape, dtype)."""
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        for name, (shape, dtype) in tensors.items():
            setattr(self, name, mx.zeros(shape, dtype=dtype))
        self._group = None
        self._idx = 0
        self._ready = False
        self._primed = None

    # -- runtime state, derived once from the loaded tensors -------------------------------------
    def _prepare(self):
        if self._ready:
            return
        w16 = self.trellis.shape[-1]
        if w16 % 16:
            raise NotImplementedError(f"half-integer EXL3 bitrate ({w16 / 16} bpw) is not supported yet")
        self._K = w16 // 16
        self._cb = 2 if "mul1" in self else 1 if "mcg" in self else 0
        suh = self["suh"] if "suh" in self else _unpack_signs(self["su"])
        svh = self["svh"] if "svh" in self else _unpack_signs(self["sv"])
        self._suh = suh.astype(mx.float16).reshape(-1)
        self._svh = svh.astype(mx.float16).reshape(-1)
        self._suh2d = self._suh.reshape(1, -1)
        self._tr = self.trellis.view(mx.uint32).reshape(-1)
        self._plans = {}
        mx.eval(self._suh, self._suh2d, self._svh, self._tr)
        self._ready = True

    @property
    def rot_suh(self) -> mx.array:
        """(1, in_features): this linear's input scales, as rotation kernels take them."""
        self._prepare()
        return self._suh2d

    def prime(self, x: mx.array, xh: mx.array):
        """The next call with exactly this array uses xh as its rotated input (fusion.py)."""
        self._primed = (x, xh)

    def gemv_rotated(self, xh: mx.array, out_dtype: mx.Dtype) -> mx.array:
        """(rows, out_features) from an already rotated input xh, (1, rows, k) or (rows, k)."""
        self._prepare()
        rows = xh.shape[-2]
        plan = self._plans.get((rows, out_dtype))
        if plan is None:
            plan = self._plans[(rows, out_dtype)] = TrellisGemvPlan(
                self.in_features, [self.out_features], self._K, self._cb, self._svh, rows, out_dtype)
        return plan(xh, [self._tr])

    def _bias(self, y):
        return y + self["bias"].astype(y.dtype) if "bias" in self else y

    def _prefill(self, xh: mx.array) -> mx.array:
        n, k = self.out_features, self.in_features
        parts = []
        for c0 in range(0, n, PREFILL_COL_CHUNK):
            c = min(PREFILL_COL_CHUNK, n - c0)
            wt = decode_wt(self._tr, k, n, self._K, self._cb, nt0=c0 // 16, ntc=c // 16)
            parts.append(xh @ wt.T)
        y = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        return _had128(y) * self._svh

    def __call__(self, x: mx.array) -> mx.array:
        if self._group is not None:
            return self._group(x)[self._idx]
        self._prepare()
        in_shape, dtype = x.shape, x.dtype
        rows = x.size // self.in_features
        if rows <= MAX_GEMV_ROWS:
            primed, self._primed = self._primed, None
            xh = primed[1] if primed is not None and primed[0] is x else \
                rot(x.reshape(rows, self.in_features), self.rot_suh)
            y = self.gemv_rotated(xh, dtype)
        else:
            xh = _had128(x.reshape(rows, self.in_features).astype(mx.float16) * self._suh)
            y = self._prefill(xh).astype(dtype)
        return self._bias(y).reshape(in_shape[:-1] + (self.out_features,))


class EXL3SiblingGroup:
    """Same-input EXL3 siblings computed in one launch (rows <= MAX_GEMV_ROWS)."""

    def __init__(self, members: list[EXL3Linear]):
        for m in members:
            m._prepare()
        first = members[0]
        self.members = members
        self.k = first.in_features
        self.ns = [m.out_features for m in members]
        self.K, self.cb = first._K, first._cb
        self.suh = mx.stack([m._suh for m in members])                  # (n_members, k)
        self.svh = mx.concatenate([m._svh for m in members])
        self.trs = [m._tr for m in members]
        self.plans: dict[tuple, TrellisGemvPlan] = {}
        self._cache_x = None
        self._cache_out = None
        self._primed = None
        mx.eval(self.suh, self.svh)
        for i, m in enumerate(members):
            m._group, m._idx = self, i

    @staticmethod
    def fusable(members: list[EXL3Linear]) -> bool:
        for m in members:
            m._prepare()
        f = members[0]
        return (2 <= len(members) <= 3
                and all(m.in_features == f.in_features and m._K == f._K and m._cb == f._cb for m in members)
                and all(m.out_features % 128 == 0 and "bias" not in m for m in members))

    @property
    def rot_suh(self) -> mx.array:
        return self.suh

    def prime(self, x: mx.array, xh: mx.array):
        """The next call with exactly this array uses xh (n_members, rows, k) as its rotated input."""
        self._primed = (x, xh)

    def gemv(self, x: mx.array, out_dtype: mx.Dtype) -> mx.array:
        """All members' outputs, concatenated: (rows, sum(ns)) in out_dtype (rows <= MAX_GEMV_ROWS)."""
        rows = x.size // self.k
        primed, self._primed = self._primed, None
        xh = primed[1] if primed is not None and primed[0] is x else rot(x.reshape(rows, self.k), self.suh)
        plan = self.plans.get((rows, out_dtype))
        if plan is None:
            plan = self.plans[(rows, out_dtype)] = TrellisGemvPlan(self.k, self.ns, self.K, self.cb, self.svh,
                                                                   rows, out_dtype)
        return plan(xh, self.trs)

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        if self._cache_x is x:
            return self._cache_out
        in_shape, dtype = x.shape, x.dtype
        rows = x.size // self.k
        if rows > MAX_GEMV_ROWS:   # prefill: each member on its own path
            outs = []
            for m in self.members:
                m._group = None
                try:
                    outs.append(m(x))
                finally:
                    m._group = self
            out = tuple(outs)
        else:
            y = self.gemv(x, dtype)
            splits = [sum(self.ns[:i + 1]) for i in range(len(self.ns) - 1)]
            out = tuple(o.reshape(in_shape[:-1] + (n,)) for o, n in zip(mx.split(y, splits, axis=-1), self.ns))
        self._cache_x, self._cache_out = x, out
        return out


class RowExactLinear(nn.Module):
    """Dense nn.Linear whose multi-row (speculative verify) results are bitwise those of single-row
    decode: MLX picks different matmul kernels for 1 vs several rows, so a verify step would otherwise
    round differently from plain decoding and MTP greedy output would drift from plain greedy.
    Rows 2..MAX_GEMV_ROWS run as independent single-row products."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        if "bias" in linear:
            self.bias = linear.bias

    def __call__(self, x: mx.array) -> mx.array:
        rows = x.size // x.shape[-1]
        w = self.weight.T
        if 1 < rows <= MAX_GEMV_ROWS:
            x2 = x.reshape(rows, x.shape[-1])
            y = mx.concatenate([x2[i:i + 1] @ w for i in range(rows)], axis=0).reshape(x.shape[:-1] + (w.shape[1],))
        else:
            y = x @ w
        return y + self.bias if "bias" in self else y
