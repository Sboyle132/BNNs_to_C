/* bnn_bconv_neon.c - CANDIDATE binary-conv kernels for the sweep. Same
 * signature as bconv_xnor / bconv_xnor_packed (drop-in): inputs are a[Cin,H,W]
 * in {-1,+1} int8 and packed_w (kidx-packed, as pack_weights produces); output
 * is integer P[Cout,Hout,Wout]. Single-thread (no OpenMP): matches the board.
 *
 * Techniques (daBNN / Larq Compute Engine), applied to the A53:
 *  - channel-packed layout: input packed ONCE (no per-pixel gather), unlike the
 *    dense bconv_xnor.
 *  - mask-free disagreement count: D = popcount(a ^ w); invalid tail channels
 *    are 0 in both -> XOR 0 -> contribute nothing, so NO per-word tail mask.
 *    P = N_valid - 2*D, N_valid = (#valid taps)*Cin.
 *  - deferred reduction (NEON): vcntq_u8 + vpadalq_u8 into uint16x8, reduced to
 *    a scalar once per output, instead of a horizontal reduce per 64-bit word.
 *  - _b4: register-tiling over 4 output channels; each activation patch word is
 *    loaded once and reused across the 4 channels (4 accumulators).
 *
 * The scalar fallback (non-aarch64, e.g. this x86 dev box) computes the SAME
 * integers, so correctness is validated here and the NEON path under qemu.
 * Verified bit-exactly against bconv_naive_int (the oracle). */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "bnn_react_arena.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

static inline int od(int in, int k, int pad, int stride) {
    return (in + 2 * pad - k) / stride + 1;
}

/* ---- shared packing (identical layout to bconv_xnor_packed) ---- */
static uint64_t *pack_input(const int8_t *a, int Cin, int H, int W, int Wc) {
    uint64_t *apack = rs_calloc((size_t)H * W * Wc * sizeof(uint64_t));
    if (!apack) { fprintf(stderr, "OOM apack\n"); exit(1); }
    for (int ci = 0; ci < Cin; ++ci) {
        int wo = ci >> 6; uint64_t bit = (uint64_t)1 << (ci & 63);
        const int8_t *ac = a + (size_t)ci * H * W;
        for (int p = 0; p < H * W; ++p)
            if (ac[p] > 0) apack[(size_t)p * Wc + wo] |= bit;
    }
    return apack;
}
static uint64_t *pack_weights_ch(const uint64_t *packed_w, int Cout, int Cin,
                                 int kh, int kw, int Wc) {
    int nwords = (Cin * kh * kw + 63) / 64;
    uint64_t *wpack = rs_calloc((size_t)Cout * kh * kw * Wc * sizeof(uint64_t));
    if (!wpack) { fprintf(stderr, "OOM wpack\n"); exit(1); }
    for (int co = 0; co < Cout; ++co) {
        const uint64_t *src = packed_w + (size_t)co * nwords;
        for (int ci = 0; ci < Cin; ++ci)
            for (int ky = 0; ky < kh; ++ky)
                for (int kx = 0; kx < kw; ++kx) {
                    int kidx = (ci * kh + ky) * kw + kx;
                    if ((src[kidx >> 6] >> (kidx & 63)) & 1) {
                        size_t tap = (size_t)(co * kh + ky) * kw + kx;
                        wpack[tap * Wc + (ci >> 6)] |= (uint64_t)1 << (ci & 63);
                    }
                }
    }
    return wpack;
}

/* ---- deferred-reduction disagreement count over nw u64 words ---- */
static inline uint32_t xor_popcnt(const uint64_t *a, const uint64_t *b, int nw) {
#ifdef __aarch64__
    uint16x8_t acc = vdupq_n_u16(0);
    int i = 0;
    for (; i + 2 <= nw; i += 2) {
        uint8x16_t x = veorq_u8(vld1q_u8((const uint8_t *)(a + i)),
                                vld1q_u8((const uint8_t *)(b + i)));
        acc = vpadalq_u8(acc, vcntq_u8(x));
    }
    uint32_t d = vaddvq_u32(vpaddlq_u16(acc));
    for (; i < nw; ++i) d += __builtin_popcountll(a[i] ^ b[i]);
    return d;
#else
    uint32_t d = 0;
    for (int i = 0; i < nw; ++i) d += __builtin_popcountll(a[i] ^ b[i]);
    return d;
#endif
}

/* shared scalar column: P for channels [c0,c1) at output (oy,ox); handles any
 * kx clip, used by the edge/tail paths of the tiled template. */
void bconv_col_scalar(const uint64_t *apack, const uint64_t *wpack, int Wc,
                      int Cin, int kh, int kw, int H, int W, int Hout, int Wout,
                      int oy, int oyb, int ky0, int ky1, int ox, int stride,
                      int pad, int32_t *P, int c0, int c1) {
    (void)H;
    int oxb = ox * stride - pad;
    int kx0 = oxb < 0 ? -oxb : 0, kx1 = (oxb + kw > W) ? W - oxb : kw;
    int vt = (ky1 - ky0) * (kx1 - kx0) * Cin;
    for (int co = c0; co < c1; ++co) {
        const uint64_t *wco = wpack + (size_t)(co * kh) * kw * Wc;
        uint32_t D = 0;
        for (int ky = ky0; ky < ky1; ++ky) {
            int iy = oyb + ky;
            for (int kx = kx0; kx < kx1; ++kx)
                D += xor_popcnt(apack + (size_t)(iy * W + oxb + kx) * Wc,
                                wco + (size_t)(ky * kw + kx) * Wc, Wc);
        }
        P[((size_t)co * Hout + oy) * Wout + ox] = vt - 2 * (int)D;
    }
}

void bconv_col_scalar_i16(const uint64_t *apack, const uint64_t *wpack, int Wc,
                      int Cin, int kh, int kw, int H, int W, int Hout, int Wout,
                      int oy, int oyb, int ky0, int ky1, int ox, int stride,
                      int pad, int16_t *P, int c0, int c1) {
    (void)H;
    int oxb = ox * stride - pad;
    int kx0 = oxb < 0 ? -oxb : 0, kx1 = (oxb + kw > W) ? W - oxb : kw;
    int vt = (ky1 - ky0) * (kx1 - kx0) * Cin;
    for (int co = c0; co < c1; ++co) {
        const uint64_t *wco = wpack + (size_t)(co * kh) * kw * Wc;
        uint32_t D = 0;
        for (int ky = ky0; ky < ky1; ++ky) {
            int iy = oyb + ky;
            for (int kx = kx0; kx < kx1; ++kx)
                D += xor_popcnt(apack + (size_t)(iy * W + oxb + kx) * Wc,
                                wco + (size_t)(ky * kw + kx) * Wc, Wc);
        }
        P[((size_t)co * Hout + oy) * Wout + ox] = (int16_t)(vt - 2 * (int)D);
    }
}

/* ===== candidate 1: channel-packed, deferred reduction, one co at a time ===== */
void bconv_packed_neon(const int8_t *a, const uint64_t *packed_w, int32_t *P,
                       int Cin, int H, int W, int Cout, int kh, int kw,
                       int pad, int stride) {
    int Hout = od(H, kh, pad, stride), Wout = od(W, kw, pad, stride);
    int Wc = (Cin + 63) / 64;
    uint64_t *apack = pack_input(a, Cin, H, W, Wc);
    uint64_t *wpack = pack_weights_ch(packed_w, Cout, Cin, kh, kw, Wc);

    for (int oy = 0; oy < Hout; ++oy) {
        int oyb = oy * stride - pad;
        int ky0 = oyb < 0 ? -oyb : 0, ky1 = (oyb + kh > H) ? H - oyb : kh;
        for (int ox = 0; ox < Wout; ++ox) {
            int oxb = ox * stride - pad;
            int kx0 = oxb < 0 ? -oxb : 0, kx1 = (oxb + kw > W) ? W - oxb : kw;
            int valid_total = (ky1 - ky0) * (kx1 - kx0) * Cin;
            for (int co = 0; co < Cout; ++co) {
                const uint64_t *wco = wpack + (size_t)(co * kh) * kw * Wc;
                uint32_t D = 0;
                for (int ky = ky0; ky < ky1; ++ky) {
                    int iy = oyb + ky;
                    for (int kx = kx0; kx < kx1; ++kx) {
                        const uint64_t *ap = apack + (size_t)(iy * W + oxb + kx) * Wc;
                        const uint64_t *wp = wco + (size_t)(ky * kw + kx) * Wc;
                        D += xor_popcnt(ap, wp, Wc);
                    }
                }
                P[((size_t)co * Hout + oy) * Wout + ox] = valid_total - 2 * (int)D;
            }
        }
    }
    rs_free(apack); rs_free(wpack);
}

/* ===== register-tiled family (template: bnn_bconv_blk.h) ===== */
#define CB 4
#define PB 1
#define SFX _b4
#include "bnn_bconv_blk.h"
#define CB 8
#define PB 1
#define SFX _b8
#include "bnn_bconv_blk.h"
#define CB 16
#define PB 1
#define SFX _b16
#include "bnn_bconv_blk.h"
#define CB 4
#define PB 2
#define SFX _b4p2
#include "bnn_bconv_blk.h"
#define CB 8
#define PB 2
#define SFX _b8p2
#include "bnn_bconv_blk.h"
#define CB 4
#define PB 4
#define SFX _b4p4
#include "bnn_bconv_blk.h"

/* int16-P variant of the engine's bconv (P fits int16: max |P|=Cin*9<=3456) */
#define CB 4
#define PB 4
#define SFX _b4p4_i16
#define BK_PT int16_t
#define BK_PCOL bconv_col_scalar_i16
#include "bnn_bconv_blk.h"

/* vector reduce+store variants (b4p4 only): 4 pixels reduced per instruction
 * group, one vector store per channel instead of 4 scalar reductions+stores. */
#define CB 4
#define PB 4
#define SFX _b4p4v
#define BK_VRED 1
#define BK_VST4(p, v) vst1q_s32((p), (v))
#include "bnn_bconv_blk.h"
#define CB 4
#define PB 4
#define SFX _b4p4v_i16
#define BK_PT int16_t
#define BK_PCOL bconv_col_scalar_i16
#define BK_VRED 1
#define BK_VST4(p, v) vst1_s16((p), vmovn_s32(v))
#include "bnn_bconv_blk.h"

/* Wc==1 pixel-pair NEON popcount + vector reduce/store (b4p4 only) */
#define CB 4
#define PB 4
#define SFX _b4p4w
#define BK_VRED 1
#define BK_WC1V 1
#define BK_VST4(p, v) vst1q_s32((p), (v))
#include "bnn_bconv_blk.h"
#define CB 4
#define PB 4
#define SFX _b4p4w_i16
#define BK_PT int16_t
#define BK_PCOL bconv_col_scalar_i16
#define BK_VRED 1
#define BK_WC1V 1
#define BK_VST4(p, v) vst1_s16((p), vmovn_s32(v))
#include "bnn_bconv_blk.h"
