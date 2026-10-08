# Notices

This project is licensed under the Apache License 2.0 (see LICENSE).

## oMLX

`scripts/install.sh` downloads [oMLX](https://github.com/jundot/omlx) (Apache License 2.0) and adds this
project to it: the `omlx/exl3` package and tools in `src/` are new files, and
`patches/omlx-exl3-hooks.patch` modifies two oMLX source files (`omlx/utils/model_loading.py`,
`omlx/cluster/worker_shim.py`).

## ExLlamaV3 / EXL3

The kernels in `src/omlx/exl3/kernels.py` decode the EXL3 format defined by ExLlamaV3 (tensor layout,
trellis bitstream and codebook constants), and `src/tools/exl3_kld.py` follows the measurement method
of ExLlamaV3's `eval/qbench`. No ExLlamaV3 source code is included. ExLlamaV3's license:

```
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Models

No model weights are included. EXL3 checkpoints (for example `turboderp/Qwen3.8-27B-exl3`) and the
original Qwen3.8 weights are distributed by their authors under their own licenses.
