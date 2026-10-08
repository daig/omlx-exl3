# SPDX-License-Identifier: Apache-2.0
"""Metal kernels for EXL3 (ExLlamaV3 trellis) linears on Apple GPUs.

Format (unchanged ExLlamaV3 layout): a K-bit tensor stores int16 trellis[k/16, n/16, 16K]; each
16x16 tile is a tail-biting bitstream of 256*K bits, read as 8K little-endian uint32 words with bits
MSB-first. Position t's 16-bit state is stream bits [(t+1)K-16, (t+1)K). Codebooks: 3inst (0),
mcg (1), mul1 (2). Positions 32g..32g+31 of a tile ("run" g) cover output columns g and 8+g of the
tile and all 16 k of the slice, from stream words [Kg, Kg+K) plus the word before.

Kernels:
  gemv   decode / speculative verify (rows <= 8): y = had128(xh @ W_inner) * svh, one dispatch,
         written in the caller's dtype. A threadgroup owns one 128-column block (2 column halves x 4 k-splits of
         simdgroups); each lane decodes one run per k-slice. Partials are reduced in a fixed order
         (bitwise deterministic, and row-exact: row r gives the same bits for any row count); k may
         also split across threadgroups, the last to arrive summing the published partials.
         Sibling groups (q/k/v, gate/up, in_proj_qkv/z) run as one launch over up to 3 member
         trellises (no copy), each member with its own rotated input.
  decode trellis -> W_inner^T (n, k) fp16 for prefill (then MLX's matmul).
  rot / norm_rot / swiglu_rot   decode-path input rotations xh = had128(x * suh) / sqrt(128) for
         1-3 members, alone, fused with the RMSNorm that produces x, or fused with SwiGLU.
"""
from __future__ import annotations

import mlx.core as mx

HEADER = r"""
#define K_INV  0.0067672729f    // fp16 0x1eee
#define K_BIAS (-10.3828125f)   // fp16 0xc931

inline half2 pairsum_6400(uint x)              // (1024 + b0 + b1, 1024 + b2 + b3) as fp16
{
    return as_type<half2>(((x & 0x00ff00ffu) + ((x >> 8) & 0x00ff00ffu)) | 0x64006400u);
}

template <int CB>
inline half dec1(uint s)
{
    if (CB == 0 || CB == 1) {
        uint h = CB == 0 ? s * 89226354u + 64248484u : s * 0xCBAC1FEDu;
        half2 v = as_type<half2>((h & 0x8fff8fffu) ^ 0x3b603b60u);
        return v.x + v.y;
    } else {
        uint h = s * 0x83DCD12Du;
        uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
        half r = as_type<half>(ushort(ushort(t) + ushort(t >> 16) + ushort(0x6400)));
        return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931)));
    }
}
"""


def _kk(p: int) -> int:
    """k inside the 16-slice for position p of a 32-run (independent of g)."""
    jj = (p % 8) % 4
    return 2 * (p // 8) + (jj % 2) + 8 * (jj // 2)


def gen_windows(K: int) -> tuple[list[str], list[str]]:
    """16-bit state source expressions for positions 0..31 of a run (words wp, w0..w{K-1})."""
    srcs, mids = [], set()
    for p in range(32):
        e = (p + 1) * K
        ih = (e - 1) // 32
        s = (ih + 1) * 32 - e
        if s <= 16:
            srcs.append(f"extract_bits(w{ih}, {s}u, 16u)" if s else f"(w{ih} & 0xffffu)")
        else:
            prev = f"w{ih - 1}" if ih > 0 else "wp"
            mids.add((ih, prev))
            srcs.append(f"extract_bits(m{ih}, {s - 16}u, 16u)")
    mid_lines = [f"uint m{ih} = insert_bits(w{ih} >> 16, {prev}, 16u, 16u);" for ih, prev in sorted(mids)]
    return mid_lines, srcs


def gen_loads(K: int) -> list[str]:
    """This lane's K stream words of the current tile (w0..) and the word before (wp)."""
    if K % 4 == 0:
        load = [f"uint4 q{v} = ((const device uint4*) tp)[{K // 4} * g + {v}];" for v in range(K // 4)]
        load += [f"uint w{4 * v + c} = q{v}.{'xyzw'[c]};" for v in range(K // 4) for c in range(4)]
    elif K % 2 == 0:
        load = [f"uint2 q{v} = ((const device uint2*) tp)[{K // 2} * g + {v}];" for v in range(K // 2)]
        load += [f"uint w{2 * v + c} = q{v}.{'xy'[c]};" for v in range(K // 2) for c in range(2)]
    else:
        load = [f"uint w{c} = tp[{K} * g + {c}];" for c in range(K)]
    load.append("uint wp = tp[wprev];")
    return load


def gen_prefetch(K: int) -> tuple[list[str], list[str], list[str]]:
    """Software-pipelined stream-word loads for the GEMV's k-slice loop: (before the loop: declare and
    load slice 0; loop top: take this slice's words and issue the next slice's loads; unpack). With
    only ~10-20 k-slices per simdgroup, a load that is consumed right away leaves the DRAM latency
    exposed on every slice; issuing the next slice's loads before decoding this one hides it."""
    if K % 4 == 0:
        loads = [(f"q{v}", "uint4", f"((const device uint4*) P)[{K // 4} * g + {v}]") for v in range(K // 4)]
        unpack = [f"uint w{4 * v + c} = q{v}.{'xyzw'[c]};" for v in range(K // 4) for c in range(4)]
    elif K % 2 == 0:
        loads = [(f"q{v}", "uint2", f"((const device uint2*) P)[{K // 2} * g + {v}]") for v in range(K // 2)]
        unpack = [f"uint w{2 * v + c} = q{v}.{'xy'[c]};" for v in range(K // 2) for c in range(2)]
    else:
        loads = [(f"w{c}", "uint", f"P[{K} * g + {c}]") for c in range(K)]
        unpack = []
    loads.append(("wp", "uint", "P[wprev]"))
    pre = [f"{t} n{v} = {t}(0);" for v, t, _ in loads]
    pre.append("if (myn > 0) { " + " ".join(f"n{v} = {e.replace('P', 'tp')};" for v, _, e in loads) + " }")
    top = [f"{t} {v} = n{v};" for v, t, _ in loads]
    top.append("if (i + 1 < myn) { const device uint* tn = tp + tstride; "
               + " ".join(f"n{v} = {e.replace('P', 'tn')};" for v, _, e in loads) + " }")
    return pre, top, unpack


# ------------------------------------------------------------------------------------------- GEMV

_GEMV_SOURCE = r"""
    constexpr uint TW = 8 * BITS;
    constexpr uint WS = 4;                                  // k-splits per threadgroup (x 2 column halves)
    const uint lane = thread_index_in_simdgroup;
    const uint warp = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;        // 128-column output block
    const uint kblock = threadgroup_position_in_grid.y;
    const uint KT = params[0], NT = params[1], TGK = params[2], K = params[3];
    const uint N = NT * 16;
    const uint half_ = warp & 1u, split = warp >> 1;
    const uint g = lane & 7;
    const uint nt = blk * 8 + half_ * 4 + (lane >> 3);

    const uint S = TGK * WS;
    const uint chunk = (KT + S - 1) / S;
    const uint ks0 = (kblock * WS + split) * chunk;
    const uint myn = ks0 < KT ? min(chunk, KT - ks0) : 0u;

    ACC_DECL

    // member of a sibling group: its own trellis buffer, tile offset and n-tile count
    const uint sub = NSUB > 1 ? tsub[nt] : 0u;
    const device uint* tbase = sub == 0 ? t0 : (sub == 1 ? t1 : t2);
    const uint NTs = params[4 + sub], ntl = nt - params[8 + sub];
    const uint wprev = (BITS * g + TW - 1) % TW;
    const device uint* tp = tbase + (ulong(ks0) * NTs + ntl) * TW;
    const ulong tstride = ulong(NTs) * TW;
    const device half4* xp = (const device half4*)(xh) + sub * MR * (K / 4) + ks0 * 4;

    PRELOAD
    for (uint i = 0; i < myn; ++i) {
        LOAD_WORDS
        X_DECL
        RUN_CODE
        tp += tstride;
        xp += 4;
    }

    // ---- k-splits -> 128-column block partial (fixed order) ----
    threadgroup float red[WS][2][32][2 * MR];
    threadgroup float bv[MR][128];
    threadgroup uint last;
    STORE_RED
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < MR * 128; idx += 256) {
        const uint r = idx >> 7, c = idx & 127;
        const uint h = c >> 6, l = ((c & 63) >> 4) * 8 + (c & 7), hl = (c >> 3) & 1;
        float v = 0.0f;
        for (uint j = 0; j < WS; ++j) v += red[j][h][l][2 * r + hl];
        bv[r][c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- k-blocks across threadgroups: the last arriver sums the published partials (fixed order) ----
    if (TGK > 1) {
        device atomic<float>* part = (device atomic<float>*) partials;
        device atomic_uint* ctr = (device atomic_uint*) counters;
        for (uint idx = tid; idx < MR * 128; idx += 256) {
            const uint r = idx >> 7, c = idx & 127;
            atomic_store_explicit(&part[(ulong(kblock) * MR + r) * N + blk * 128 + c], bv[r][c], memory_order_relaxed);
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0)
            last = atomic_fetch_add_explicit(&ctr[blk], 1u, memory_order_relaxed) == TGK - 1 ? 1u : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (last == 0) return;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        for (uint idx = tid; idx < MR * 128; idx += 256) {
            const uint r = idx >> 7, c = idx & 127;
            float v = 0.0f;
            for (uint kb = 0; kb < TGK; ++kb)
                v += atomic_load_explicit(&part[(ulong(kb) * MR + r) * N + blk * 128 + c], memory_order_relaxed);
            bv[r][c] = v;
        }
        if (tid == 0) atomic_store_explicit(&ctr[blk], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- 128-point FWHT (Sylvester order), * 1/sqrt(128) * svh -> OT ----
    for (uint hs = 1; hs < 128; hs <<= 1) {
        for (uint idx = tid; idx < MR * 64; idx += 256) {
            const uint r = idx >> 6, b = idx & 63;
            const uint i0 = (b / hs) * (2 * hs) + (b % hs), i1 = i0 + hs;
            const float a0 = bv[r][i0], a1 = bv[r][i1];
            bv[r][i0] = a0 + a1;
            bv[r][i1] = a0 - a1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint idx = tid; idx < MR * 128; idx += 256) {
        const uint r = idx >> 7, c = idx & 127;
        y[ulong(r) * N + blk * 128 + c] = OT(bv[r][c] * 0.08838834764831845f * float(svh[blk * 128 + c]));
    }
"""


def _gemv_source(K: int, mr: int) -> str:
    mid_lines, srcs = gen_windows(K)
    run = list(mid_lines)
    for p in range(32):
        half_ = "lo" if (p % 8) < 4 else "hi"
        fmas = " ".join(f"a{r}_{half_} = fma(float(d), x{r}_{_kk(p)}, a{r}_{half_});" for r in range(mr))
        run.append(f"{{ half d = dec1<CB>({srcs[p]}); {fmas} }}")
    xdecl = []
    for r in range(mr):
        for q in range(4):
            xdecl.append(f"float4 xq{r}_{q} = float4(xp[{q} + {r} * (K / 4)]);")
        for kk in range(16):
            xdecl.append(f"float x{r}_{kk} = xq{r}_{kk // 4}.{'xyzw'[kk % 4]};")
    acc = "\n    ".join(f"float a{r}_lo = 0.0f, a{r}_hi = 0.0f;" for r in range(mr))
    store = "\n    ".join(f"red[split][half_][lane][{2 * r}] = a{r}_lo; red[split][half_][lane][{2 * r + 1}] = a{r}_hi;"
                          for r in range(mr))
    pre, top, unpack = gen_prefetch(K)
    return (_GEMV_SOURCE.replace("ACC_DECL", acc)
            .replace("PRELOAD", "\n    ".join(pre))
            .replace("LOAD_WORDS", "\n        ".join(top + unpack))
            .replace("X_DECL", "\n        ".join(xdecl))
            .replace("RUN_CODE", "\n        ".join(run))
            .replace("STORE_RED", store))


_gemv_kernels: dict = {}


def _gemv_kernel(K: int, mr: int):
    key = (K, mr)
    if key not in _gemv_kernels:
        _gemv_kernels[key] = mx.fast.metal_kernel(
            name=f"omlx_exl3_gemv_k{K}_m{mr}",
            input_names=["xh", "t0", "t1", "t2", "params", "tsub", "svh", "counters"],
            output_names=["y", "partials"], header=HEADER, source=_gemv_source(K, mr))
    return _gemv_kernels[key]


def pick_tgk(KT: int, blocks: int, ws: int = 4, target_tgs: int = 512, min_slices: int = 8) -> int:
    """Split k across threadgroups until there are >= target_tgs threadgroups (~8 per core), keeping
    >= min_slices k-slices per k-split. Under layer-after-layer execution a layer's tail wave idles
    the GPU; on Qwen3.8-27B shapes ~8+ threadgroups per core beat ~2 by 10-25%."""
    tgk = 1
    while blocks * tgk < target_tgs and KT // (ws * tgk * 2) >= min_slices:
        tgk *= 2
    return tgk


MAX_GEMV_ROWS = 8


class TrellisGemvPlan:
    """y = had128(xh @ W_inner) * svh (out_dtype) for one EXL3 linear or a sibling group of <= 3
    members sharing K and codebook. xh is (n_members, rows, k): each member's own rotated input."""

    def __init__(self, k: int, member_ns: list[int], K: int, cb: int, svh: mx.array, rows: int,
                 out_dtype: mx.Dtype = mx.float16):
        assert k % 16 == 0 and all(n % 128 == 0 for n in member_ns) and 1 <= K <= 8 and cb in (0, 1, 2)
        assert 1 <= len(member_ns) <= 3 and 1 <= rows <= MAX_GEMV_ROWS
        self.k, self.n, self.K, self.cb, self.rows = k, sum(member_ns), K, cb, rows
        self.out_dtype = out_dtype
        self.nsub = len(member_ns)
        self.KT, self.NT = k // 16, self.n // 16
        self.blocks = self.n // 128
        self.tgk = pick_tgk(self.KT, self.blocks)
        nts = [n // 16 for n in member_ns]
        offs = [sum(nts[:i]) for i in range(len(nts))]
        pad = lambda v: v + [0] * (3 - len(v))  # noqa: E731
        self.params = mx.array([self.KT, self.NT, self.tgk, k] + pad(nts) + [0] + pad(offs) + [0], dtype=mx.uint32)
        self.tsub = mx.array(sum(([i] * nt for i, nt in enumerate(nts)), []) if self.nsub > 1 else [0] * 8,
                             dtype=mx.uint32)
        self.svh = svh.astype(mx.float16).reshape(-1)
        # Per-block arrival counters, zero at rest (the last arriver resets its block): no fill per call.
        # >= 64 entries so MLX binds the buffer in the device (writable) address space.
        self.counters = mx.zeros((max(self.blocks, 64),), dtype=mx.uint32)
        self.kern = _gemv_kernel(K, rows)
        mx.eval(self.params, self.tsub, self.svh, self.counters)

    def __call__(self, xh: mx.array, trellises: list[mx.array]) -> mx.array:
        """trellises: flat uint32 views, one per member. Returns (rows, n) in out_dtype."""
        t = list(trellises) + [trellises[0]] * (3 - len(trellises))
        return self.kern(
            inputs=[xh, t[0], t[1], t[2], self.params, self.tsub, self.svh, self.counters],
            template=[("BITS", self.K), ("CB", self.cb), ("MR", self.rows), ("NSUB", self.nsub),
                      ("OT", self.out_dtype)],
            grid=(self.blocks * 256, self.tgk, 1), threadgroup=(256, 1, 1),
            output_shapes=[(self.rows, self.n), (self.tgk, self.rows, self.n) if self.tgk > 1 else (1,)],
            output_dtypes=[self.out_dtype, mx.float32])[0]


# ----------------------------------------------------------------------------------------- decode

_DECODE_SOURCE = r"""
    constexpr uint TW = 8 * BITS;
    const uint gid = thread_position_in_grid.x;
    const uint KT = params[0], NT = params[1], nt0 = params[2], NTC = params[3];
    const uint g = gid & 7;
    const uint run = gid >> 3;                              // (kt, local n-tile)
    const uint ntl = run % NTC, kt = run / NTC;
    if (kt >= KT) return;
    const uint nt = nt0 + ntl;
    const uint wprev = (BITS * g + TW - 1) % TW;
    const device uint* tp = trellis + (ulong(kt) * NT + nt) * TW;
    LOAD_WORDS
    half lo[16], hi[16];
    DECODE_CODE
    const ulong Ktot = ulong(KT) * 16;
    device half4* wlo = (device half4*)(w + (ulong(ntl) * 16 + g) * Ktot + kt * 16);
    device half4* whi = (device half4*)(w + (ulong(ntl) * 16 + 8 + g) * Ktot + kt * 16);
    for (uint q = 0; q < 4; ++q) {
        wlo[q] = half4(lo[4 * q], lo[4 * q + 1], lo[4 * q + 2], lo[4 * q + 3]);
        whi[q] = half4(hi[4 * q], hi[4 * q + 1], hi[4 * q + 2], hi[4 * q + 3]);
    }
"""


def _decode_source(K: int) -> str:
    mid_lines, srcs = gen_windows(K)
    code = list(mid_lines)
    for p in range(32):
        code.append(f"{'lo' if (p % 8) < 4 else 'hi'}[{_kk(p)}] = dec1<CB>({srcs[p]});")
    return (_DECODE_SOURCE.replace("LOAD_WORDS", "\n    ".join(gen_loads(K)))
            .replace("DECODE_CODE", "\n    ".join(code)))


_decode_kernels: dict = {}


def decode_wt(trellis_u32: mx.array, k: int, n: int, K: int, cb: int, nt0: int = 0, ntc: int | None = None) -> mx.array:
    """W_inner^T columns [16*nt0, 16*(nt0+ntc)) as an (16*ntc, k) fp16 matrix (rows = output features)."""
    ntc = n // 16 - nt0 if ntc is None else ntc
    if K not in _decode_kernels:
        _decode_kernels[K] = mx.fast.metal_kernel(
            name=f"omlx_exl3_decode_k{K}", input_names=["trellis", "params"], output_names=["w"],
            header=HEADER, source=_decode_source(K))
    params = mx.array([k // 16, n // 16, nt0, ntc], dtype=mx.uint32)
    return _decode_kernels[K](
        inputs=[trellis_u32, params], template=[("BITS", K), ("CB", cb)],
        grid=((k // 16) * ntc * 8, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(ntc * 16, k)], output_dtypes=[mx.float16])[0]


# ------------------------------------------------------------------------------- input rotations
# Rotated inputs are member-major, xh (NM, rows, K) fp16, so each consumer's slice is contiguous.
# One threadgroup of 128 threads per (128-block, row), one element per thread: FWHT stages 1-16 by
# simd_shuffle_xor, 32 and 64 through threadgroup memory, all in fp32. Shapes are template constants.

_ROT_HEADER = r"""
// sum of squares of 8 packed 16-bit floats (bf16 when BF, else fp16)
template <bool BF> inline float sumsq8(uint4 v) {
    float s = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float a, b;
        if (BF) { a = as_type<float>(v[i] << 16); b = as_type<float>(v[i] & 0xffff0000u); }
        else { const half2 h = as_type<half2>(v[i]); a = float(h.x); b = float(h.y); }
        s = fma(a, a, s);
        s = fma(b, b, s);
    }
    return s;
}
"""

_FWHT = r"""
    for (ushort h = 1; h < 32; h <<= 1) {
        const float p = simd_shuffle_xor(v, h);
        v = (lane & h) ? p - v : v + p;
    }
    for (ushort h = 32; h < 128; h <<= 1) {
        tg[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float p = tg[tid ^ h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v = (tid & h) ? p - v : v + p;
    }
"""

_ROT_PRE = r"""
    const uint tid = thread_position_in_threadgroup.x, lane = tid & 31;
    const uint blk = threadgroup_position_in_grid.x, r = threadgroup_position_in_grid.y;
    const uint ROWS = threadgroups_per_grid.y;
    const uint c = blk * 128 + tid;
    threadgroup float tg[128];
"""

_ROTATE_MEMBERS = r"""
    for (uint m = 0; m < NM; ++m) {
        float v = xv * float(suh[m * KD + c]);
        FWHT
        xh[(ulong(m) * ROWS + r) * KD + c] = half(v * 0.08838834764831845f);
    }
""".replace("FWHT", _FWHT)

_ROT_SOURCE = _ROT_PRE + r"""
    const float xv = float(x[ulong(r) * KD + c]);
""" + _ROTATE_MEMBERS

# RMSNorm with MLX's rounding (normalize in fp32, round to T, scale by the weight, round to T): the
# norm output other consumers read matches mx.fast.rms_norm's, and EXL3 members see that value.
# The row's sum of squares: 16-byte loads, KD / 1024 per thread, all issued before use.
_NORM_ROT_SOURCE = _ROT_PRE + r"""
    threadgroup float part[4];
    const device T* xr = x + ulong(r) * KD;
    const device uint4* xv4 = (const device uint4*)(xr);
    float ss = 0.0f;
    #pragma unroll
    for (uint i = tid; i < KD / 8; i += 128) ss += sumsq8<BF>(xv4[i]);
    ss = simd_sum(ss);
    if (lane == 0) part[tid >> 5] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt((part[0] + part[1] + part[2] + part[3]) / float(KD) + eps[0]);
    const T xn_t = T(float(w[c]) * float(T(float(xr[c]) * inv)));
    xn[ulong(r) * KD + c] = xn_t;
    const float xv = float(xn_t);
""" + _ROTATE_MEMBERS

# gu = [gate | up] (rows, 2 KD) -> had128(silu(gate) * up * suh) / sqrt(128), (1, rows, KD)
_SWIGLU_ROT_SOURCE = _ROT_PRE + r"""
    const float g = float(gu[ulong(r) * 2 * KD + c]), u = float(gu[ulong(r) * 2 * KD + KD + c]);
    float v = g / (1.0f + metal::precise::exp(-g)) * u * float(suh[c]);
    FWHT
    xh[ulong(r) * KD + c] = half(v * 0.08838834764831845f);
""".replace("FWHT", _FWHT)

_rot_kernels: dict = {}
_eps_arrays: dict = {}


def _rot_kernel(name: str):
    if name not in _rot_kernels:
        src, ins, outs = {
            "rot": (_ROT_SOURCE, ["x", "suh"], ["xh"]),
            "norm_rot": (_NORM_ROT_SOURCE, ["x", "w", "suh", "eps"], ["xn", "xh"]),
            "swiglu_rot": (_SWIGLU_ROT_SOURCE, ["gu", "suh"], ["xh"]),
        }[name]
        _rot_kernels[name] = mx.fast.metal_kernel(name=f"omlx_exl3_{name}", input_names=ins, output_names=outs,
                                                  header=_ROT_HEADER, source=src)
    return _rot_kernels[name]


def rot(x: mx.array, suh: mx.array) -> mx.array:
    """x (rows, K) any float dtype, suh (NM, K) fp16 -> xh (NM, rows, K) fp16."""
    rows, k = x.shape
    nm = suh.shape[0]
    return _rot_kernel("rot")(
        inputs=[x, suh], template=[("KD", k), ("NM", nm)],
        grid=(k, rows, 1), threadgroup=(128, 1, 1),
        output_shapes=[(nm, rows, k)], output_dtypes=[mx.float16])[0]


def norm_rot(x: mx.array, w: mx.array, eps: float, suh: mx.array) -> tuple[mx.array, mx.array]:
    """RMSNorm(x) * w in x.dtype (as mx.fast.rms_norm rounds it) and its rotations xh (NM, rows, K) fp16."""
    rows, k = x.shape
    nm = suh.shape[0]
    assert k % 1024 == 0 and x.dtype in (mx.bfloat16, mx.float16)
    e = _eps_arrays.get(eps)
    if e is None:
        e = _eps_arrays[eps] = mx.array([eps], dtype=mx.float32)
    xn, xh = _rot_kernel("norm_rot")(
        inputs=[x, w, suh, e], template=[("T", x.dtype), ("BF", x.dtype == mx.bfloat16), ("KD", k), ("NM", nm)],
        grid=(k, rows, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows, k), (nm, rows, k)], output_dtypes=[x.dtype, mx.float16])
    return xn, xh


def swiglu_rot(gu: mx.array, suh: mx.array) -> mx.array:
    """gu (rows, 2N) = [gate | up], suh (N,) fp16 -> had128(silu(gate) * up * suh) / sqrt(128), (1, rows, N) fp16."""
    rows, n2 = gu.shape
    n = n2 // 2
    return _rot_kernel("swiglu_rot")(
        inputs=[gu, suh], template=[("KD", n)],
        grid=(n, rows, 1), threadgroup=(128, 1, 1),
        output_shapes=[(1, rows, n)], output_dtypes=[mx.float16])[0]
