"""Smoke test: load an EXL3 checkpoint through omlx.exl3 (mlx-vlm), greedy-generate, report speed."""
import logging
import sys
import time

import mlx.core as mx
from mlx_vlm import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template

from omlx.exl3 import load
from omlx.exl3.linear import EXL3Linear

logging.basicConfig(level=logging.INFO, format="%(message)s")
path = sys.argv[1]
n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128
prompt_text = "Explain in a few paragraphs how a hash table handles collisions, with a short Python example."

t0 = time.perf_counter()
model, processor = load(path)
n_exl3 = sum(1 for _, m in model.named_modules() if isinstance(m, EXL3Linear))
print(f"loaded in {time.perf_counter() - t0:.1f}s, {n_exl3} EXL3Linear modules")

prompt = apply_chat_template(processor, model.config, prompt_text, num_images=0)
text, last = "", None
t0 = time.perf_counter()
for resp in stream_generate(model, processor, prompt, max_tokens=n_tokens, temperature=0.0):
    text += resp.text
    last = resp
print(text[:600])
print(f"\nprompt {last.prompt_tokens} tok @ {last.prompt_tps:.1f} tok/s | generation {last.generation_tokens} tok @ "
      f"{last.generation_tps:.1f} tok/s | peak mem {last.peak_memory:.1f} GB")
