# SPDX-License-Identifier: Apache-2.0
"""Load ExLlamaV3 (EXL3) checkpoints through mlx-vlm with EXL3Linear layers.

The checkpoint's EXL3 tensors (<prefix>.trellis / suh / svh / su / sv / mul1 / mcg / bias) are
inventoried from the safetensors headers; the model class's __init__ is wrapped so that each
nn.Linear whose (sanitized) path has an inventory entry becomes an EXL3Linear declaring exactly
those tensors; generic quantization config is hidden from mlx-vlm; then mlx-vlm's normal strict
load runs (tokenizer, processor, vision tower and all fp16 tensors load as usual). Afterwards,
same-input siblings with equal K and codebook are grouped into single-launch EXL3SiblingGroups and
the decode-path rotation fusions are installed (fusion.py).

Supported now: integer bitrates (1-8 bpw), codebooks 3inst / mcg / mul1, mlx-vlm qwen3_5 (Qwen3.5 /
3.6 / 3.8 dense), including the checkpoint's MTP head when oMLX's MTP patch keeps it (its tensors
bind under language_model.mtp.*). Not yet: half-integer mul1 rates, MoE expert tensors, text-only
(mlx-lm) loading.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import struct
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .fusion import fuse_decode
from .linear import EXL3Linear, EXL3SiblingGroup, RowExactLinear

logger = logging.getLogger(__name__)

_DTYPES = {"I16": mx.int16, "F16": mx.float16, "BF16": mx.bfloat16, "F32": mx.float32, "I32": mx.int32,
           "U16": mx.uint16, "U32": mx.uint32}
_EXL3_PARTS = ("trellis", "suh", "svh", "su", "sv", "mul1", "mcg", "bias")
_SIBLINGS = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"), ("in_proj_qkv", "in_proj_z"))

_active_inventory: dict[str, dict[str, tuple[tuple[int, ...], Any]]] | None = None
_patched_classes: set = set()


def is_exl3_config(config: dict[str, Any]) -> bool:
    qc = config.get("quantization_config") or {}
    return str(qc.get("quant_method", "")).lower() == "exl3"


def _read_header(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return header


def inventory(model_path: Path) -> dict[str, dict[str, tuple[tuple[int, ...], Any]]]:
    """EXL3 linear prefix (checkpoint naming) -> {part: (shape, dtype)}."""
    index = model_path / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() \
        else ["model.safetensors"]
    headers: dict[str, Any] = {}
    for f in files:
        headers.update(_read_header(model_path / f))
    inv: dict[str, dict[str, tuple[tuple[int, ...], Any]]] = {}
    for key in headers:
        if key.endswith(".trellis"):
            prefix = key[: -len(".trellis")]
            inv[prefix] = {part: (tuple(headers[f"{prefix}.{part}"]["shape"]), _DTYPES[headers[f"{prefix}.{part}"]["dtype"]])
                           for part in _EXL3_PARTS if f"{prefix}.{part}" in headers}
    return inv


def _sanitize_key_fn(model_class):
    import importlib
    module = importlib.import_module(model_class.__module__)
    return getattr(module, "sanitize_key", lambda k: k)


def _module_paths(prefix: str, sanitize_key) -> list[str]:
    """Candidate module paths for a checkpoint prefix. oMLX's MTP pre-load patch keeps mtp.* weights
    under language_model.mtp.*, which mlx-vlm's sanitize_key does not rename."""
    paths = [sanitize_key(prefix)]
    if prefix.startswith("mtp."):
        paths.append("language_model." + prefix)
    return paths


def _swap_linears(model: nn.Module, inv: dict[str, dict[str, Any]], sanitize_key) -> int:
    by_path = {path: parts for prefix, parts in inv.items() for path in _module_paths(prefix, sanitize_key)}
    leaves = dict(tree_flatten(model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)))
    swapped = 0
    for path, module in list(leaves.items()):
        parts = by_path.get(path)
        if parts is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"EXL3 tensor at {path} targets a {type(module).__name__}, expected Linear")
        out_features, in_features = module.weight.shape
        leaves[path] = EXL3Linear(in_features, out_features, parts)
        swapped += 1
    model.update_modules(tree_unflatten(list(leaves.items())))
    return swapped


def _patch_model_class(model_class) -> None:
    if model_class in _patched_classes:
        return
    original_init = model_class.__init__
    sanitize_key = _sanitize_key_fn(model_class)

    def patched_init(self, config):
        original_init(self, config)
        if _active_inventory is not None:
            n = _swap_linears(self, _active_inventory, sanitize_key)
            logger.info("EXL3: bound %d trellis linears", n)

    model_class.__init__ = patched_init
    _patched_classes.add(model_class)


def row_exact_dense(model: nn.Module) -> int:
    """Wrap the language model's remaining dense nn.Linear layers (Qwen3.5: in_proj_a/b) so
    speculative-verify rows match single-row decode bit for bit (see RowExactLinear)."""
    leaves = dict(tree_flatten(model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)))
    wrapped = 0
    for path, module in list(leaves.items()):
        if path.startswith("language_model.") and type(module) is nn.Linear:
            leaves[path] = RowExactLinear(module)
            wrapped += 1
    if wrapped:
        model.update_modules(tree_unflatten(list(leaves.items())))
    return wrapped


def group_siblings(model: nn.Module) -> int:
    """Group same-input EXL3 siblings (q/k/v, gate/up, in_proj_qkv/z) into single-launch groups. One
    launch needs one bitrate, so siblings are partitioned by (bitrate, codebook): self-calibrated
    quants give q its own rate but k and v a shared one, which still saves a launch."""
    groups = 0
    for _, module in model.named_modules():
        for names in _SIBLINGS:
            members = [getattr(module, n, None) for n in names]
            if not all(isinstance(m, EXL3Linear) for m in members):
                continue
            for m in members:
                m._prepare()
            by_rate: dict = {}
            for m in members:
                by_rate.setdefault((m._K, m._cb), []).append(m)
            for subset in by_rate.values():
                if EXL3SiblingGroup.fusable(subset):
                    EXL3SiblingGroup(subset)
                    groups += 1
    return groups


def unify_float_dtypes(model: nn.Module) -> int:
    """Cast non-EXL3 floating parameters to the embedding dtype. EXL3 checkpoints store unquantized
    tensors in fp16 and mlx-vlm casts most (not all: Qwen3.5's in_proj_a/b stay fp16) to bf16; mixed
    fp16 x bf16 matmuls promote to fp32, which breaks oMLX's verify-width GDN kernel and slows decode.
    EXL3Linear computes in fp16 internally and returns the caller's dtype, so it is left alone."""
    target = None
    for name, m in model.named_modules():
        if name.endswith("embed_tokens") and "weight" in m:
            target = m["weight"].dtype
            break
    if target is None:
        return 0
    exl3 = {name for name, m in model.named_modules() if isinstance(m, EXL3Linear)}
    updates = []
    for k, v in tree_flatten(model.parameters()):
        owner = k.rsplit(".", 1)[0]
        if owner in exl3 or not mx.issubdtype(v.dtype, mx.floating) or v.dtype == target:
            continue
        updates.append((k, v.astype(target)))
    if updates:
        model.update(tree_unflatten(updates))
    return len(updates)


def load(model_name: str | Path, *, is_vlm: bool = True):
    """(model, processor) for an EXL3 checkpoint, via mlx-vlm's public loader."""
    global _active_inventory
    if not is_vlm:
        raise NotImplementedError("EXL3 text-only (mlx-lm) loading is not wired up yet; load as a VLM")
    model_path = Path(model_name)
    inv = inventory(model_path)
    if not inv:
        raise ValueError(f"no EXL3 trellis tensors found in {model_path}")

    import mlx_vlm.utils as vlm_utils
    from mlx_vlm.models.qwen3_5 import qwen3_5

    _patch_model_class(qwen3_5.Model)
    original_load_config = vlm_utils.load_config

    def load_config_without_quantization(path, *args, **kwargs):
        loaded = copy.deepcopy(original_load_config(path, *args, **kwargs))
        for cfg in (loaded, loaded.get("text_config") or {}):
            cfg.pop("quantization", None)
            cfg.pop("quantization_config", None)
        return loaded

    _active_inventory = inv
    vlm_utils.load_config = load_config_without_quantization
    try:
        logger.info("EXL3: loading %s (%d trellis linears)", model_path, len(inv))
        model, processor = vlm_utils.load(model_path, trust_remote_code=False, strict=True)
    finally:
        vlm_utils.load_config = original_load_config
        _active_inventory = None
    cast = unify_float_dtypes(model)
    groups = group_siblings(model)
    # Off by default: oMLX's own multi-row verify path is not row-exact either (MTP greedy text already
    # differs from plain greedy on stock oMLX), so this only moves the divergence and costs ~4% decode.
    dense = row_exact_dense(model) if os.environ.get("OMLX_EXL3_ROW_EXACT_DENSE") == "1" else 0
    # Decode-path input-rotation fusion into the producing norm / SwiGLU (OMLX_EXL3_FUSE=0 disables)
    fused = fuse_decode(model) if os.environ.get("OMLX_EXL3_FUSE", "1") != "0" else {}
    logger.info("EXL3: %d sibling groups, %d float tensors cast to the activation dtype, %d row-exact dense "
                "linears, fused rotations %s", groups, cast, dense, fused)
    return model, processor
