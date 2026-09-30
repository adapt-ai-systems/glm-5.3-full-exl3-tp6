// exl3mm stage 1: FC2 (down projection) + finalize for EXL3 TR3 experts, GB10 sm_121a.
//
// Our own code. Trellis layout/decode semantics learned from exllamav3 (3INST cb=1 "mcg"
// codebook, mult 0xCBAC1FED) and validated against the E3 down kernel by output comparison.
//
// fc2 : ybuf[r, :] = fp16( had128(fp16(inter[r] @ Wd_e)) / sqrt(128) * svh_e * route_weight[r] )
//       per route row r (expert-sorted order), no atomics.
// finalize : out[t] = bf16( sum_k ybuf[pos[t*8+k]] ) fp32 accumulate, fixed k order (deterministic).
//
// Expert down trellis layout: (K/16=32, tiles_n=384, 16*bits uint16 words), each 16x16 tile is a
// tail-biting MSB-first stream of 256 16-bit windows; lane<<3 windows decode directly to the
// mma.m16n8k16 B fragments (cols 0-7, cols 8-15).

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

constexpr int K = 512;                       // intermediate size per rank (fc2 reduction dim)
constexpr int N = 6144;                      // hidden size (fc2 output dim)
constexpr int TILES_N16 = N / 16;            // 384 16-col blocks
constexpr int THREADS = 512;                 // 2 ping-pong groups of 8 warps
constexpr int GTHREADS = 256;
constexpr int GROUPS = 2;
constexpr int GWARPS = 8;
constexpr int TILE_N = GWARPS * 16;          // 128 columns per group tile (one warp = one 16-col block)
constexpr int NTILES = N / TILE_N;           // 48
constexpr int MB = 4;                        // 16-row mma blocks -> 64 rows per segment
constexpr int A_ROW_BYTES = K * 2;           // 1024
constexpr int A_BYTES = MB * 16 * A_ROW_BYTES;
constexpr int STAGES = 4;
constexpr int KS16 = 2;                      // k16 steps per pipeline stage (k32)
constexpr int KSTEPS = K / (16 * KS16);      // 16 stages per tile
constexpr int B_STAGE_BYTES = KS16 * GWARPS * 128; // K4 worst case
constexpr int SC_STRIDE = 136;               // fp32 staging row stride (conflict-free float2 stores)
constexpr int SC_BYTES = 16 * SC_STRIDE * 4;
constexpr int SMEM_BYTES = A_BYTES + GROUPS * (STAGES * B_STAGE_BYTES + SC_BYTES);
constexpr float HAD_SCALE = 0.088388347648f;  // 1/sqrt(128)
#ifndef STAGGER_NS
#define STAGGER_NS 2000
#endif

__device__ __forceinline__ void group_sync(int g)
{
    asm volatile("bar.sync %0, %1;\n" :: "r"(1 + g), "n"(GTHREADS) : "memory");
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* smem)
{
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 3INST mcg codebook: two 16-bit trellis windows -> half2 (w0, w1)
__device__ __forceinline__ uint32_t cb2(uint32_t w0, uint32_t w1)
{
    uint32_t x0 = w0 * 0xCBAC1FEDu, x1 = w1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
    x1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    __half2 r = __hadd2(*reinterpret_cast<__half2*>(&lo), *reinterpret_cast<__half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// Per-lane decode constants for a tile stored as bits*8 uint32 words
template <int BITS>
struct LaneDec
{
    int i0, i2, s2;
    __device__ __forceinline__ void init(int lane)
    {
        const int b1 = (lane * 8 + 257) * BITS;
        const int b0 = b1 - 16;
        const int b2 = b1 + BITS * 7;
        i0 = (b0 / 32) % (BITS * 8);
        i2 = ((b2 - 1) / 32) % (BITS * 8);
        s2 = (((b2 - 1) / 32) + 1) * 32 - b2;
    }
    // tile: bits*8 uint32 words; out: B fragments for cols 0-7 (f0[2]) and cols 8-15 (f1[2])
    __device__ __forceinline__ void decode(const uint32_t* tile, uint32_t (&f0)[2], uint32_t (&f1)[2]) const
    {
        const uint32_t a = tile[i0], b = tile[i2];
        const uint64_t m = (static_cast<uint64_t>(a) << 32) | b;
        uint32_t w[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = static_cast<uint32_t>(m >> (s2 + BITS * (7 - j))) & 0xffffu;
        f0[0] = cb2(w[0], w[1]);
        f0[1] = cb2(w[2], w[3]);
        f1[0] = cb2(w[4], w[5]);
        f1[1] = cb2(w[6], w[7]);
    }
};

// 128-point Hadamard (natural order) of a row held as 4 consecutive floats per lane, scaled by 1/sqrt(128)
__device__ __forceinline__ void had128(float (&v)[4], int lane)
{
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1; v[1] = d0 + d1; v[2] = s0 - s1; v[3] = d0 - d1;
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1)
    {
        const float f = (lane & m) ? -1.f : 1.f;
        #pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const float p = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = fmaf(v[e], f, p);
        }
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e) v[e] *= HAD_SCALE;
}

__device__ __forceinline__ float rnd_half(float x) { return __half2float(__float2half_rn(x)); }

template <int BITS>
__device__ __forceinline__ void fc2_body(
    const half* __restrict__ inter, const uint16_t* __restrict__ down, const half* __restrict__ svh,
    half* __restrict__ ybuf, const float* __restrict__ row_weight,
    int row0, int rows, unsigned char* smem)
{
    constexpr int W = 16 * BITS;                 // uint16 words per 16x16 tile
    constexpr int TILE_BYTES = W * 2;
    constexpr int CPT = TILE_BYTES / 16;         // 16-byte chunks per tile (8 for K4, 6 for K3)
    constexpr int B_CHUNKS = KS16 * GWARPS * CPT;

    const int t = threadIdx.x;
    const int grp = t >> 8;                      // ping-pong group
    const int tg = t & (GTHREADS - 1);
    const int warp = tg >> 5, lane = t & 31;     // warp within group
    unsigned char* sA = smem;
    unsigned char* sB = smem + A_BYTES + grp * (STAGES * B_STAGE_BYTES);
    float* sC = reinterpret_cast<float*>(smem + A_BYTES + GROUPS * STAGES * B_STAGE_BYTES + grp * SC_BYTES);

    const int tiles_per_cta = NTILES / gridDim.x;
    const int tile0 = blockIdx.x * tiles_per_cta + grp;   // this group takes every other tile
    const int ntl = tiles_per_cta / GROUPS;
    const int S = ntl * KSTEPS;

    auto load_b = [&](int slot, int s)
    {
        if (tg < B_CHUNKS)
        {
            const int tile = tile0 + GROUPS * (s / KSTEPS), ks = s % KSTEPS;
            const int j = tg / (GWARPS * CPT);
            const int rem = tg % (GWARPS * CPT);
            const int cbi = rem / CPT, q = rem % CPT;
            const uint16_t* src = down + (static_cast<size_t>(ks * KS16 + j) * TILES_N16 + tile * GWARPS + cbi) * W + q * 8;
            cp_async16(sB + slot * B_STAGE_BYTES + tg * 16, src);
        }
    };

    // A rows resident, loaded by the whole CTA
    for (int c = t; c < rows * (A_ROW_BYTES / 16); c += THREADS)
    {
        const int r = c >> 6, ch = c & 63;
        cp_async16(sA + r * A_ROW_BYTES + ((ch ^ (r & 7)) << 4), inter + static_cast<size_t>(row0 + r) * K + ch * 8);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    // Phase-shift the two groups: without this they run in lockstep and their epilogues coincide
    // (measured 4.3 ms vs 2.8 ms at M=4096); the exact delay is irrelevant (0..8 us all give the same time).
    if (grp == 1) __nanosleep(STAGGER_NS);

    #pragma unroll
    for (int p = 0; p < STAGES - 1; ++p)
    {
        if (p < S) load_b(p, p);
        cp_commit();
    }

    LaneDec<BITS> ld;
    ld.init(lane);

    float acc[MB][2][4];
    #pragma unroll
    for (int mb = 0; mb < MB; ++mb)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[mb][h][e] = 0.f;

    const int a_lane_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int a_lane_chunk = lane >> 4;

    for (int s = 0; s < S; ++s)
    {
        cp_wait<STAGES - 2>();
        group_sync(grp);
        if (s + STAGES - 1 < S) load_b((s + STAGES - 1) % STAGES, s + STAGES - 1);
        cp_commit();

        const int ks = s % KSTEPS;
        const unsigned char* sb = sB + (s % STAGES) * B_STAGE_BYTES;
        #pragma unroll
        for (int j = 0; j < KS16; ++j)
        {
            uint32_t fb[2][2];
            const uint32_t* tile = reinterpret_cast<const uint32_t*>(sb + (j * GWARPS + warp) * TILE_BYTES);
            ld.decode(tile, fb[0], fb[1]);
            const int kchunk = (ks * KS16 + j) * 2 + a_lane_chunk;
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                if (mb * 16 < rows)
                {
                    uint32_t fa[4];
                    const int r = mb * 16 + a_lane_row;
                    ldsm_x4(fa, sA + r * A_ROW_BYTES + ((kchunk ^ (r & 7)) << 4));
                    mma16816(acc[mb][0], fa, fb[0][0], fb[0][1]);
                    mma16816(acc[mb][1], fa, fb[1][0], fb[1][1]);
                }
            }
        }

        if (ks == KSTEPS - 1)
        {
            const int tile = tile0 + GROUPS * (s / KSTEPS);
            const int n_base = tile * TILE_N;
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                if (mb * 16 < rows)
                {
                    #pragma unroll
                    for (int h = 0; h < 2; ++h)
                    {
                        const int col = warp * 16 + h * 8 + 2 * (lane & 3);
                        const int r0 = lane >> 2;
                        *reinterpret_cast<float2*>(sC + r0 * SC_STRIDE + col) = make_float2(acc[mb][h][0], acc[mb][h][1]);
                        *reinterpret_cast<float2*>(sC + (r0 + 8) * SC_STRIDE + col) = make_float2(acc[mb][h][2], acc[mb][h][3]);
                    }
                    group_sync(grp);
                    #pragma unroll
                    for (int rr = 0; rr < 2; ++rr)
                    {
                        const int r = warp + rr * GWARPS;
                        const int lrow = mb * 16 + r;
                        if (lrow < rows)
                        {
                            const float w = row_weight[row0 + lrow];
                            float4 x = *reinterpret_cast<const float4*>(sC + r * SC_STRIDE + lane * 4);
                            float v[4] = {rnd_half(x.x), rnd_half(x.y), rnd_half(x.z), rnd_half(x.w)};
                            had128(v, lane);
                            const uint2 sraw = *reinterpret_cast<const uint2*>(svh + n_base + lane * 4);
                            const float2 s01 = __half22float2(*reinterpret_cast<const __half2*>(&sraw.x));
                            const float2 s23 = __half22float2(*reinterpret_cast<const __half2*>(&sraw.y));
                            const __half2 o01 = __floats2half2_rn(v[0] * s01.x * w, v[1] * s01.y * w);
                            const __half2 o23 = __floats2half2_rn(v[2] * s23.x * w, v[3] * s23.y * w);
                            uint2 o;
                            o.x = *reinterpret_cast<const uint32_t*>(&o01);
                            o.y = *reinterpret_cast<const uint32_t*>(&o23);
                            *reinterpret_cast<uint2*>(ybuf + static_cast<size_t>(row0 + lrow) * N + n_base + lane * 4) = o;
                        }
                    }
                    group_sync(grp);
                }
            }
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) acc[mb][h][e] = 0.f;
        }
    }
    cp_wait<0>();
}

}  // namespace

extern "C" __global__ __launch_bounds__(THREADS, 1)
void exl3mm_fc2(
    const half* __restrict__ inter,                    // [R, 512]
    const uint16_t* const* __restrict__ down_ptrs,     // per expert
    const half* const* __restrict__ svh_ptrs,          // per expert [6144]
    const int* __restrict__ expert_bits,               // per expert 3|4
    const int* __restrict__ seg_expert,
    const int* __restrict__ seg_row0,
    const int* __restrict__ seg_rows,
    const int* __restrict__ num_segs_ptr,
    const float* __restrict__ row_weight,              // [R]
    half* __restrict__ ybuf)                           // [R, 6144]
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int seg = blockIdx.y;
    if (seg >= *num_segs_ptr) return;
    const int e = seg_expert[seg];
    const int row0 = seg_row0[seg];
    const int rows = seg_rows[seg];
    if (rows <= 0) return;
    if (expert_bits[e] == 3)
        fc2_body<3>(inter, down_ptrs[e], svh_ptrs[e], ybuf, row_weight, row0, rows, smem_raw);
    else
        fc2_body<4>(inter, down_ptrs[e], svh_ptrs[e], ybuf, row_weight, row0, rows, smem_raw);
}

// One CTA per token: out[t] = bf16(sum_k ybuf[pos[t*8+k]]), fixed k order.
extern "C" __global__ __launch_bounds__(256)
void exl3mm_finalize(
    const half* __restrict__ ybuf,        // [R, 6144]
    const int* __restrict__ pos,          // [M*8], -1 = route dropped (non-local expert)
    const int* __restrict__ num_rows_ptr,
    __nv_bfloat16* __restrict__ out)      // [M, 6144]
{
    constexpr int CH = N / 8;             // 768 chunks of 8 halves
    const int tok = blockIdx.x;
    const int nr = *num_rows_ptr;
    int p[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        const int v = pos[tok * 8 + k];
        p[k] = (v >= 0 && v < nr) ? v : -1;
    }
    for (int c = threadIdx.x; c < CH; c += blockDim.x)
    {
        uint4 raw[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            raw[k] = make_uint4(0, 0, 0, 0);
            if (p[k] >= 0) raw[k] = __ldcs(reinterpret_cast<const uint4*>(ybuf + static_cast<size_t>(p[k]) * N + c * 8));
        }
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            const uint32_t words[4] = {raw[k].x, raw[k].y, raw[k].z, raw[k].w};
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&words[i]));
                acc[2 * i] += f.x;
                acc[2 * i + 1] += f.y;
            }
        }
        uint4 o;
        uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const __nv_bfloat162 b = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
            ow[i] = *reinterpret_cast<const uint32_t*>(&b);
        }
        *reinterpret_cast<uint4*>(out + static_cast<size_t>(tok) * N + c * 8) = o;
    }
}
