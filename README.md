# EXL3 for oMLX: 8-bit quality in 6 bits on Apple Silicon

This repository adds the EXL3 quantization format from [ExLlamaV3](https://github.com/turboderp-org/exllamav3) to [oMLX](https://github.com/jundot/omlx), the MLX-based LLM server for Apple Silicon. It includes Metal kernels written for the format.

EXL3 fits a model into far fewer bits than standard MLX quantization at the same quality. On a Mac, fewer bits means faster generation and more free memory.

On an M5 Ultra, Qwen3.8-27B in EXL3 6.0 bpw **matches MLX 8-bit quality**, while using **22% less memory** and **decoding 16% faster**. With the multi-token prediction (MTP) head that ships in the checkpoint, it decodes **2.3× faster**.

| Qwen3.8-27B, M5 Ultra | MLX 8-bit | **EXL3 6.0 bpw (this repo)** |
|---|---|---|
| KL divergence from BF16, in-domain chat (lower is better) | 0.00068 | 0.00083 |
| KL divergence from BF16, held-out text | 0.0075 | 0.0078 (statistical tie) |
| Top-1 token agreement with BF16 | 99.3% | 99.1% |
| Weights (decoder + output head) | 25.4 GiB | **17.9 GiB** |
| Memory when loaded | 27.9 GB | **21.8 GB** |
| Decode | 33.6 tok/s | **39.1 tok/s** |
| Decode with MTP (depth 3) | n/a: the checkpoint has no MTP head | **78.0 tok/s** (66–90 depending on the text) |
| Prefill (4.6K-token prompt) | **1,355 tok/s** | 1,172 tok/s |

For scale, BF16 compared against itself with rounding-level noise scores 0.00048 in-domain and 0.0053 on held-out text, so both quants sit close to that floor. Full methodology and numbers: [results/quality.md](results/quality.md) and [results/speed.md](results/speed.md).

## Why EXL3

- **On a Mac, bits per weight are speed and memory.** Generating a token reads every weight once, so decode speed is roughly memory bandwidth divided by model size, and the model has to fit next to its context cache. What matters is how much quality each bit buys.
- **Standard quantization spends bits inefficiently.** MLX's built-in 4-, 6- and 8-bit quants round each weight to an evenly spaced grid with a scale for every small group of weights. GGUF's Q-formats work much the same way. This is simple and fast to decode, but at 4–6 bits it loses more quality than it needs to. That's why MLX 8-bit, 8.5 bits per weight including scales, is the usual near-lossless choice.
- **EXL3 gets much closer to the limit.** turboderp's EXL3 format is based on [QTIP](https://arxiv.org/abs/2406.11235):
  - It rotates each weight matrix with randomized Hadamard transforms so the values look Gaussian.
  - It encodes them with a trellis code: a compact bitstream that a cheap hash function turns back into weights.
  - It allocates bits per tensor. The "SC" (self-calibrated) checkpoints spend bits where the model is most sensitive.
- **What it compares to.**
  - **EXL3 6.0 bpw matches MLX 8-bit quality** at 71% of the weight size.
  - **MLX 6-bit-class quants are about the size of EXL3 6.0** but diverge from BF16 2–3× more. oMLX's oQ6 at 6.56 bpw measures KL divergence 0.0023 against 0.00083 for EXL3 6.0.
  - **EXL3 5.0 bpw is as good as oQ6** at 77% of its size.
- **Why it needed new kernels.** ExLlamaV3 targets NVIDIA GPUs. Decoding trellis-coded weights takes about ten arithmetic operations per weight, so a direct port is compute-bound on Apple GPUs and ends up slower than MLX 8-bit. The kernels here decode, multiply and rotate in a single pass, which keeps generation memory-bound.

## Quick start

**Requirements:**
- An Apple Silicon Mac with enough memory. EXL3 6.0 uses about 22 GB plus the context cache. Tested on an M5 Ultra with 256 GB, macOS 27.
- Xcode command line tools.
- git.
- Python 3.12, found automatically via `uv` or `python3.12`.
- The Hugging Face CLI (`hf`).

**1. Install.** This sets up oMLX at a pinned commit with EXL3 support, in its own virtualenv:

```sh
git clone https://github.com/daig/omlx-exl3.git
cd omlx-exl3
scripts/install.sh                # installs into ~/omlx-exl3; pass a directory to change it
```

**2. Download the model.** turboderp's EXL3 quants have one branch per bitrate:

```sh
hf download turboderp/Qwen3.8-27B-exl3 --revision SC_6.00bpw_H6 --local-dir models/Qwen3.8-27B-exl3-SC6
```

**3. Serve:**

```sh
scripts/serve.sh
```

- **Web chat:** http://localhost:8000/admin/chat. Pick **Qwen3.8-27B-exl3-SC6**.
- **OpenAI-compatible API:** http://localhost:8000/v1. For example:

```sh
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen3.8-27B-exl3-SC6",
  "messages": [{"role": "user", "content": "Explain how a hash table handles collisions."}],
  "temperature": 0.6, "top_p": 0.95, "top_k": 20}'
```

The first request after startup takes a few extra seconds while Metal kernels compile.

## Configuration

- **MTP.** On first run, [configs/model_settings.json](configs/model_settings.json) is copied to `~/.omlx-exl3/model_settings.json`, with MTP on at depth 3 for the 6.0 and 5.0 bpw models. Change it there and restart, or use the model settings in the dashboard at http://localhost:8000/admin. Depth 3 was the most consistent setting; see [results/speed.md](results/speed.md#mtp-depth). MTP doesn't change the output: every drafted token is verified by the full model.
- **Reasoning effort.** Qwen3.8's chat template accepts `low`, `medium` or `xhigh` (the default). Set a model default by adding `"chat_template_kwargs": {"reasoning_effort": "low"}` to its settings, or send `"reasoning_effort": "low"` with a request.
- **Smaller model.** EXL3 5.0 bpw is 15.1 GiB, close to oQ6 quality. Download `--revision SC_5.00bpw_H6` into `models/Qwen3.8-27B-exl3-SC5`.
- **Other models.** This is a complete oMLX install, so any other model folder in `models/` (MLX quants and so on) is served as well.
- **Environment variables:**
  - `OMLX_EXL3_HOME`: the install directory.
  - `MODEL_DIR`: the models folder.
  - `OMLX_BASE`: the state directory. The default `~/.omlx-exl3` keeps this install apart from any other oMLX.
  - `PORT`: the server port.
  - `OMLX_EXL3_FUSE=0`: turns off the fused input rotations, for comparison.

## How it works

- **No expanded weights.** Each EXL3 layer keeps the checkpoint's trellis bitstreams and rotation scales; weights are never expanded in memory.
- **Generation and MTP verification** (up to 8 tokens at a time):
  - One Metal kernel per layer decodes the trellis, does the matrix-vector product, reduces across threadgroups and applies the output rotation.
  - Projections that share an input (q/k/v, gate/up, the linear-attention input projections) run in one launch.
  - The input rotation is fused into the preceding RMSNorm or SwiGLU kernel.
- **Prompt processing.** Weights are decoded to fp16 in chunks and multiplied with MLX's matrix multiply.
- **MTP.** The checkpoint's own MTP head (EXL3, 4 bpw) drafts tokens through oMLX's Lightning MTP.
- **Reproducibility.** Reductions run in a fixed order, so results are bitwise reproducible run to run.

## Check it yourself

```sh
H=~/omlx-exl3
# kernel tests against fp32 references (all bitrates 3-8, 1-8 rows)
$H/.venv/bin/python $H/tools/exl3_kernel_test.py

# quality against BF16 (needs mlx-community/Qwen3.8-27B-bf16, ~51 GB; the trace ships with the EXL3 model)
$H/.venv/bin/python $H/tools/exl3_kld.py --ref /path/to/Qwen3.8-27B-bf16 \
    --model SC6=models/Qwen3.8-27B-exl3-SC6 \
    --trace models/Qwen3.8-27B-exl3-SC6/qbench_prompts_gen.json --floor --out kld_trace

# decode speed outside the server (plain decode, fused vs unfused rotations)
$H/.venv/bin/python $H/tools/exl3_decode_bench.py models/Qwen3.8-27B-exl3-SC6
```

## Compatibility

- **Tested:** Qwen3.8-27B at EXL3 SC 6.0 and 5.0 bpw on an M5 Ultra.
- **Should work:** other dense Qwen3.5 / 3.6 / 3.8 EXL3 checkpoints at integer bitrates (1–8 bpw, all three EXL3 codebooks). Other Apple Silicon chips should also work, but the kernels were tuned on M5 and are untested elsewhere.
- **Not yet supported:** half-integer bitrates (for example 2.5 bpw), mixture-of-experts models, and text-only (mlx-lm) loading.
- **oMLX version:** pinned to commit `a4048eef` of oMLX main. The changes to oMLX itself are two small hooks ([patches/omlx-exl3-hooks.patch](patches/omlx-exl3-hooks.patch)); everything else is a new `omlx/exl3` package ([src/](src/)).

## Credits

- **EXL3 format and checkpoints:** turboderp, [ExLlamaV3](https://github.com/turboderp-org/exllamav3), MIT. Quality methodology after ExLlamaV3's `qbench`.
- **QTIP:** Tseng, Sun, Hou and De Sa, *QTIP: Quantization with Trellises and Incoherence Processing* (NeurIPS 2024).
- **Serving and runtime:** [oMLX](https://github.com/jundot/omlx) (Apache-2.0), [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).
- **Model:** Qwen3.8 by the Qwen team.

## License

Apache-2.0, the same as oMLX. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
