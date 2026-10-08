# Quality: KL divergence against BF16

**Model:** Qwen3.8-27B. **Reference:** `mlx-community/Qwen3.8-27B-bf16`. Measured October 2026.

## Method

- **Same code path for every model.** All models run on MLX in one process, through mlx-vlm's Qwen3.5 model code. EXL3 checkpoints use this repository's kernels.
- **Metric.** Per-token KL(BF16 ‖ model) over the full 248,077-token vocabulary, as in ExLlamaV3's `qbench`. Means come with 95% intervals from a bootstrap over rows, and top-1 agreement is the share of positions where the model's most likely token matches BF16's.
- **Two test sets:**
  - **In-domain.** turboderp's evaluation trace, which ships with the EXL3 quants: 24 multi-turn chat, reasoning and tool-use rows. It scores the 45,930 sampled response tokens, with 19,016 tokens of context. This is the trace behind turboderp's published charts for this model.
  - **Held-out.** wikitext-2 test, framed as a chat reply: 20 rows × 2,048 tokens. It is independent of the self-generated text the SC ("self-calibrated") quants were calibrated on.
- **Noise floor.** BF16 rerun with bf16-rounding-sized multiplicative noise after every layer, again as in `qbench`. It is the divergence you would see from an equivalent but different choice of kernels, so values near it are indistinguishable from "no loss".
- **Size.** Decoder layers plus output head, embeddings excluded, the convention of turboderp's charts. bpw is that size divided by the number of weights in those tensors.

Tool: `tools/exl3_kld.py`. Per-model summaries are in `kld_trace.json` and `kld_wikitext.json`.

## Results

| Model | Weights | bpw | KLD, in-domain (95% CI) | Top-1, in-domain | KLD, wikitext (95% CI) | Top-1, wikitext |
|---|---|---|---|---|---|---|
| BF16 noise floor | 47.7 GiB | 16 | 0.00048 (0.00040–0.00055) | 99.43% | 0.0053 (0.0034–0.0073) | 98.31% |
| MLX 8-bit (affine, group 64) | 25.4 GiB | 8.50 | 0.00068 (0.00055–0.00080) | 99.26% | 0.0075 (0.0054–0.0096) | 97.68% |
| **EXL3 SC 6.0 bpw** | **17.9 GiB** | **6.01** | **0.00083 (0.00067–0.00098)** | **99.13%** | **0.0078 (0.0059–0.0098)** | **97.47%** |
| oMLX oQ6 (6/8-bit mixed) | 19.6 GiB | 6.56 | 0.00230 (0.00189–0.00269) | 98.45% | 0.0158 (0.0104–0.0226) | 96.30% |
| EXL3 SC 5.0 bpw | 15.1 GiB | 5.07 | 0.00177 (0.00137–0.00219) | 98.67% | 0.0168 (0.0125–0.0218) | 96.14% |

Paired per-token comparisons, with row-bootstrap 95% intervals:

- **EXL3 SC6 vs MLX 8-bit.**
  - **In-domain:** +0.00015 (+0.00008 to +0.00022), so 8-bit is slightly closer to BF16.
  - **Wikitext:** +0.0003 (−0.0013 to +0.0019), a statistical tie.
  - In both sets the two sit close to the noise floor.
- **EXL3 SC5 vs oQ6.** SC5 is better in-domain (−0.00053) and tied on wikitext, at 77% of oQ6's size.
- **All other differences** are significant on both sets.

Perplexity moves only in the fourth decimal (in-domain: BF16 1.3335, MLX 8-bit 1.3337, SC6 1.3339), and on wikitext some quants even score below BF16. KLD is the meaningful metric at this level.

## Checks on the implementation

- **Decode path.** Generating through the KV cache with this repository's decode kernels gives the same KLD as the teacher-forced measurement on the same positions. That holds one token at a time and three at a time (MTP verification), with paired differences consistent with zero over 3,020 positions.
- **Fidelity.** Unquantized tensors load bit-identical to the BF16 checkpoint. Computing every EXL3 linear in exact fp32 instead of the shipped fp16 path changes SC6's in-domain KLD by 0.00002 (2%).
- **Against turboderp's chart.** On the same trace, turboderp's chart for the CUDA implementation shows noise floor 0.00045, SC5 0.0016 and SC6 about 0.0006. The SC quants here measure about 0.0002 higher, the same offset for both. That is consistent with the different reference engine (HF transformers on CUDA there, MLX here).
- **Clean install.** A fresh install from `scripts/install.sh` reproduces these numbers bit-for-bit.
