# SPDX-License-Identifier: Apache-2.0
"""Decode-path fusion of EXL3 input rotations into the kernels that produce their inputs.

Every EXL3 linear consumes xh = had128(x * suh) / sqrt(128). Unfused, a decode step pays a rotation
launch per linear or sibling group on top of the producer's own kernel, and on the M5 a dependent
launch costs ~4-8 us however small. For rows <= MAX_GEMV_ROWS (decode and speculative verify):

  RMSNorm -> EXL3 consumers  input_layernorm -> q/k/v or in_proj_qkv/z, post_attention_layernorm ->
                             gate/up, the final norm -> lm_head: one kernel writes the norm output
                             (other consumers such as in_proj_a/b still read it) and every consumer's
                             rotated input, grouped or not (SC quants give q/k/v different bitrates,
                             so they run as separate GEMVs but still share this kernel)
  SwiGLU -> down_proj        one kernel from the gate/up GEMV output to down_proj's rotated input

Priming is keyed on the identity of the norm's output array: a consumer that receives any other
array (a reshape, a cast) computes the rotation itself, so every call path stays correct. Modules
keep their classes' behaviour for prefill (rows > MAX_GEMV_ROWS); the fused classes are subclasses
swapped in per instance, so isinstance checks elsewhere (oMLX patches) still hold.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .kernels import MAX_GEMV_ROWS, norm_rot, swiglu_rot
from .linear import EXL3Linear

_fused_classes: dict = {}
ENABLED = True          # runtime switch (benchmarks): False makes the fused modules behave like their bases


class _Consumers:
    """The EXL3 consumers (sibling groups or single linears) of one norm output, primed together."""

    def __init__(self, consumers: list):
        self.consumers = consumers
        self.nms = [c.rot_suh.shape[0] for c in consumers]
        self.rot_suh = mx.concatenate([c.rot_suh for c in consumers]) if len(consumers) > 1 else consumers[0].rot_suh
        mx.eval(self.rot_suh)

    def prime(self, x: mx.array, xh: mx.array):
        if len(self.consumers) == 1:
            self.consumers[0].prime(x, xh)
            return
        o = 0
        for c, nm in zip(self.consumers, self.nms):
            c.prime(x, xh[o:o + nm])
            o += nm


def _fused_norm_class(base):
    if ("norm", base) not in _fused_classes:
        class FusedNorm(base):
            def __call__(self, x):
                rows = x.size // x.shape[-1]
                if rows > MAX_GEMV_ROWS or not ENABLED:
                    return base.__call__(self, x)
                target = self.__dict__["_exl3_target"]
                xn, xh = norm_rot(x.reshape(rows, x.shape[-1]), self["weight"], self.eps, target.rot_suh)
                xn = xn.reshape(x.shape)
                target.prime(xn, xh)
                return xn

        FusedNorm.__name__ = FusedNorm.__qualname__ = base.__name__
        _fused_classes[("norm", base)] = FusedNorm
    return _fused_classes[("norm", base)]


def _fused_mlp_class(base):
    if ("mlp", base) not in _fused_classes:
        class FusedMLP(base):
            def __call__(self, x):
                rows = x.size // x.shape[-1]
                if rows > MAX_GEMV_ROWS or not ENABLED:
                    return base.__call__(self, x)
                gu = self.gate_proj._group.gemv(x, mx.float16)        # (rows, 2 * hidden): [gate | up]
                xh = swiglu_rot(gu, self.down_proj._suh)
                y = self.down_proj.gemv_rotated(xh, x.dtype)
                return y.reshape(x.shape[:-1] + (y.shape[-1],))

        FusedMLP.__name__ = FusedMLP.__qualname__ = base.__name__
        _fused_classes[("mlp", base)] = FusedMLP
    return _fused_classes[("mlp", base)]


def _consumers(module, names) -> list | None:
    """Distinct EXL3 consumers (group, else the linear itself) of the named linears, in order."""
    out = []
    for n in names:
        lin = module.get(n) if module is not None else None
        if not isinstance(lin, EXL3Linear):
            return None
        c = lin._group if lin._group is not None else lin
        if all(c is not o for o in out):
            out.append(c)
    return out


def _fuse_norm(norm, consumers) -> bool:
    if not consumers or type(norm) is not nn.RMSNorm or "weight" not in norm \
            or norm["weight"].shape[0] % 1024 or norm["weight"].dtype not in (mx.bfloat16, mx.float16):
        return False
    norm.__dict__["_exl3_target"] = _Consumers(consumers)       # plain attribute: not a child module
    norm.__class__ = _fused_norm_class(type(norm))
    return True


def fuse_decode(model: nn.Module) -> dict[str, int]:
    """Install the decode-path fusions on every eligible decoder layer (and the output head)."""
    counts = {"norm": 0, "mlp": 0, "head": 0}
    for _, layer in model.named_modules():
        if not all(k in layer for k in ("input_layernorm", "post_attention_layernorm", "mlp")):
            continue
        if "self_attn" in layer:
            first = _consumers(layer["self_attn"], ("q_proj", "k_proj", "v_proj"))
        else:
            first = _consumers(layer.get("linear_attn"), ("in_proj_qkv", "in_proj_z"))
        counts["norm"] += _fuse_norm(layer.input_layernorm, first)
        mlp = layer.mlp
        second = _consumers(mlp, ("gate_proj", "up_proj"))
        counts["norm"] += _fuse_norm(layer.post_attention_layernorm, second)
        gu = mlp.gate_proj._group if second else None
        if gu is not None and len(gu.members) == 2 and gu.members[0] is mlp.gate_proj \
                and gu.members[1] is mlp.up_proj and gu.ns[0] == gu.ns[1] \
                and isinstance(mlp.get("down_proj"), EXL3Linear) and "bias" not in mlp.down_proj:
            mlp.__class__ = _fused_mlp_class(type(mlp))
            counts["mlp"] += 1
    lm = model.get("language_model", model)
    inner = lm.get("model")
    if inner is not None and "norm" in inner:
        counts["head"] += _fuse_norm(inner.norm, _consumers(lm, ("lm_head",)))
    return counts
