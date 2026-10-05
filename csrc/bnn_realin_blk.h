/* bnn_realin_blk.h - register-blocked real-input kernels, MICROBENCH CANDIDATES.
 * Template: include repeatedly with CB, PB and SFX defined; each inclusion
 * emits one pair of kernels with names suffixed by SFX.
 *   CB  : output channels per block (multiple of 4)
 *   PB  : output pixels per block, along x (multiple of 4)
 *   SFX : name suffix token, e.g. _c4p8
 *
 * Why: the current kernels re-load and re-store the output tile every input
 * channel (proj) or reduce a strict-order dot product (conv_realin). On an
 * in-order A53 that load/store traffic, not the FMAs, is the limit. Here a
 * CB x PB tile of accumulators lives in vector registers across the WHOLE
 * Cin (and tap) loop; per Cin step the only memory traffic is PB/4 input
 * vector loads and CB/4 weight vector loads for CB*PB/4 FMAs. Weights are
 * repacked once per call so each step's CB weights are one contiguous load
 * and the FMAs use the by-element form.
 *
 * Portable GNU C vector extension (v4): maps to NEON on aarch64, SSE/AVX on
 * x86. No intrinsics, so the same source is tested here and run on the board.
 * Summation order differs from the current kernels (float noise only).
 *
 *   proj_blk : x[Cin,H,W], Wm[Cout,Cin] fp32  -> out[Cout,H,W]   (1x1, no bias)
 *   conv3_blk: x[Cin,H,W], wsign[Cout,Cin,3,3] +-1 int8, alpha[Cout]
 *              -> out[Cout,H,W]  (3x3, pad 1, stride 1), out = alpha*sum
 * Requires Cout % CB == 0 (all react layers have Cout a multiple of 16). */
#ifndef RB_COMMON
#define RB_COMMON
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef float v4 __attribute__((vector_size(16), aligned(4), may_alias));
#define RB_CAT_(a, b) a##b
#define RB_CAT(a, b) RB_CAT_(a, b)
#endif

#define RB_NAME(x) RB_CAT(x, SFX)
#define RB_NV (PB / 4)
#define RB_NW (CB / 4)

static void RB_NAME(proj_blk)(const float *x, const float *Wm, float *out,
                              int Cin, int H, int W, int Cout) {
    if (Cout % CB) { fprintf(stderr, "proj_blk: Cout %% CB != 0\n"); exit(1); }
    long HW = (long)H * W;
    int ncb = Cout / CB;
    float *wt = rs_alloc((size_t)Cin * Cout * sizeof(float));
    if (!wt) { fprintf(stderr, "OOM wt\n"); exit(1); }
    for (int cb = 0; cb < ncb; ++cb)
        for (int ci = 0; ci < Cin; ++ci)
            for (int c = 0; c < CB; ++c)
                wt[((size_t)cb * Cin + ci) * CB + c] = Wm[(size_t)(cb * CB + c) * Cin + ci];

    long nfull = HW - HW % PB;
    const v4 zero = {0.f, 0.f, 0.f, 0.f};
    for (long p0 = 0; p0 < nfull; p0 += PB) {
        for (int cb = 0; cb < ncb; ++cb) {
            v4 acc[CB][RB_NV];
            for (int c = 0; c < CB; ++c)
                for (int j = 0; j < RB_NV; ++j) acc[c][j] = zero;
            const float *wp = wt + (size_t)cb * Cin * CB;
            for (int ci = 0; ci < Cin; ++ci) {
                const float *xr = x + (size_t)ci * HW + p0;
                v4 xv[RB_NV];
                for (int j = 0; j < RB_NV; ++j) xv[j] = *(const v4 *)(xr + 4 * j);
                const v4 *wv = (const v4 *)(wp + (size_t)ci * CB);
                for (int k = 0; k < RB_NW; ++k) {
                    v4 w4 = wv[k];
                    for (int cc = 0; cc < 4; ++cc)
                        for (int j = 0; j < RB_NV; ++j)
                            acc[4 * k + cc][j] += xv[j] * w4[cc];
                }
            }
            for (int c = 0; c < CB; ++c)
                for (int j = 0; j < RB_NV; ++j)
                    *(v4 *)(out + (size_t)(cb * CB + c) * HW + p0 + 4 * j) = acc[c][j];
        }
    }
    for (long p = nfull; p < HW; ++p)                 /* pixel tail, scalar */
        for (int co = 0; co < Cout; ++co) {
            float a = 0.f;
            for (int ci = 0; ci < Cin; ++ci) a += Wm[(size_t)co * Cin + ci] * x[(size_t)ci * HW + p];
            out[(size_t)co * HW + p] = a;
        }
    rs_free(wt);
}

static void RB_NAME(conv3_blk)(const float *x, const int8_t *wsign, const float *alpha,
                               float *out, int Cin, int H, int W, int Cout) {
    if (Cout % CB) { fprintf(stderr, "conv3_blk: Cout %% CB != 0\n"); exit(1); }
    long HW = (long)H * W;
    int ncb = Cout / CB;
    /* wt[cb][ci][tap][c]: the CB weights of one (ci,tap) are one contiguous run */
    float *wt = rs_alloc((size_t)Cin * 9 * Cout * sizeof(float));
    if (!wt) { fprintf(stderr, "OOM wt\n"); exit(1); }
    for (int cb = 0; cb < ncb; ++cb)
        for (int ci = 0; ci < Cin; ++ci)
            for (int t = 0; t < 9; ++t)
                for (int c = 0; c < CB; ++c)
                    wt[(((size_t)cb * Cin + ci) * 9 + t) * CB + c] =
                        (float)wsign[((size_t)(cb * CB + c) * Cin + ci) * 9 + t];

    const v4 zero = {0.f, 0.f, 0.f, 0.f};
    for (int oy = 0; oy < H; ++oy) {
        int ky0 = oy >= 1 ? 0 : 1, ky1 = oy <= H - 2 ? 3 : 2;   /* clip rows (zero pad) */
        int ox0 = 1;
        /* interior blocks: every tap column ox-1..ox+PB is in range, no clipping */
        for (; ox0 + PB + 1 <= W; ox0 += PB) {
            for (int cb = 0; cb < ncb; ++cb) {
                v4 acc[CB][RB_NV];
                for (int c = 0; c < CB; ++c)
                    for (int j = 0; j < RB_NV; ++j) acc[c][j] = zero;
                const float *wp = wt + (size_t)cb * Cin * 9 * CB;
                for (int ci = 0; ci < Cin; ++ci) {
                    for (int ky = ky0; ky < ky1; ++ky) {
                        const float *rp = x + ((size_t)ci * H + (oy + ky - 1)) * W + (ox0 - 1);
                        for (int kx = 0; kx < 3; ++kx) {
                            v4 xv[RB_NV];
                            for (int j = 0; j < RB_NV; ++j) xv[j] = *(const v4 *)(rp + kx + 4 * j);
                            const v4 *wv = (const v4 *)(wp + ((size_t)ci * 9 + ky * 3 + kx) * CB);
                            for (int k = 0; k < RB_NW; ++k) {
                                v4 w4 = wv[k];
                                for (int cc = 0; cc < 4; ++cc)
                                    for (int j = 0; j < RB_NV; ++j)
                                        acc[4 * k + cc][j] += xv[j] * w4[cc];
                            }
                        }
                    }
                }
                for (int c = 0; c < CB; ++c) {
                    int co = cb * CB + c;
                    for (int j = 0; j < RB_NV; ++j)
                        *(v4 *)(out + (size_t)co * HW + (size_t)oy * W + ox0 + 4 * j) = acc[c][j] * alpha[co];
                }
            }
        }
        /* scalar columns: ox=0, and everything from the end of the interior on */
        for (int ox = 0; ox < W; ++ox) {
            if (ox >= 1 && ox < ox0) { ox = ox0 - 1; continue; }
            for (int co = 0; co < Cout; ++co) {
                float a = 0.f;
                for (int ci = 0; ci < Cin; ++ci)
                    for (int ky = ky0; ky < ky1; ++ky)
                        for (int kx = 0; kx < 3; ++kx) {
                            int ix = ox + kx - 1;
                            if (ix < 0 || ix >= W) continue;
                            a += (float)wsign[((size_t)co * Cin + ci) * 9 + ky * 3 + kx] *
                                 x[((size_t)ci * H + (oy + ky - 1)) * W + ix];
                        }
                out[(size_t)co * HW + (size_t)oy * W + ox] = alpha[co] * a;
            }
        }
    }
    rs_free(wt);
}

#undef CB
#undef PB
#undef SFX
#undef RB_NAME
#undef RB_NV
#undef RB_NW
