# Speed

**Machine:** Mac Studio, M5 Ultra (64 GPU cores), 256 GB, macOS 27. **Software:** oMLX `a4048eef` plus this repository (fresh `scripts/install.sh`), MLX 0.32.2. Measured October 2026.

## Method

- **Serving.** Models are served by `scripts/serve.sh` and measured through oMLX's OpenAI-compatible API, one request at a time, with the GPU otherwise idle.
- **Decode tok/s.** Tokens 33 to 544 of a reply, with thinking on and temperature 0. Each figure is the best of 2 runs, on four prompt types: technical prose with code, a math word problem, code generation and creative writing.
- **Prefill tok/s.** A 4,644-token prompt divided by the time to the first token.
- **Memory.** The model's resident size as reported by oMLX after loading.

## Results

| Qwen3.8-27B | Decode (tok/s) | Prefill (tok/s) | Memory |
|---|---|---|---|
| MLX 8-bit | 33.6 | 1,355 | 27.9 GB |
| **EXL3 SC 6.0 bpw** | **39.1** (+16%) | 1,172 (−13%) | **21.8 GB** (−22%) |
| **EXL3 SC 6.0 bpw + MTP, depth 3** | **78.0** (2.3×), by prompt type: 83.3 prose+code, 90.0 math, 72.7 code, 65.8 creative | 1,152 | 21.8 GB |

- **Decode** reads every weight once per token, so it is bound by memory bandwidth. EXL3's smaller weights make it faster even though decoding each weight takes more arithmetic.
- **MTP** (multi-token prediction) drafts tokens with a small head and verifies them with the full model in one pass. It is lossless: every emitted token is one the full model accepts. The EXL3 checkpoint ships Qwen's MTP head, quantized to 4 bpw. The `mlx-community` 8-bit checkpoint does not include it, so the MTP rows are EXL3-only. Speed with MTP depends on how predictable the text is.
- **Prefill** is about 13% slower than MLX 8-bit. Long prompts are processed by decoding each EXL3 weight matrix to fp16 and using MLX's matrix multiply.

## MTP depth

Mean decode tok/s over the four prompt types (same kernels):

| MTP config | Temperature 0 | Sampled (T 0.6, top-p 0.95, top-k 20) |
|---|---|---|
| off | 39.9 | |
| depth 1 | 62.6 | |
| depth 2 | 75.4 | 76.1 |
| **depth 3 (default here)** | 77.5 | **77.9** |
| depth 4 | **78.4** | 75.7 |
| depth 5 | 70.6 | |
| adaptive (oMLX controller, max 4) | 76.2 | 75.0 |

Depth 3 is the most consistent choice. Depth 4 gains on math and code but loses on creative text. Each extra drafted token makes the verification pass about 3–4 ms more expensive.

## Other ways to run EXL3 on a Mac

Measured on the same machine on 2026-10-07 with the then-current releases, Qwen3.8-27B SC 6.0 bpw, plain decode:

- **PonyExl3:** 16.5 tok/s, or 25.8 tok/s with MTP.
- **MLXL3 v1.4.2:** 23.2 tok/s.
