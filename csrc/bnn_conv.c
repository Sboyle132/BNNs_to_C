/*
 * bnn_conv.c - binary convolution kernels producing the integer accumulator
 * P for a W1A1 conv (activations in {-1,+1}, weights sign(w) in {-1,+1}).
 *
 * Kernels (identical integer result, for A/B and dispatch):
 *   bconv_naive_int    : plain triple loop in +-1 arithmetic. The oracle.
 *   bconv_xnor         : kidx-packed XNOR+popcount (channels AND taps packed
 *                        densely into one bitstream). Re-gathers a per-pixel
 *                        im2col patch. Wins at SMALL Cin (dense words, few
 *                        popcounts, gather cheap). OpenMP over output rows.
 *   bconv_xnor_packed  : daBNN-style channel-packed layout. Input packed ONCE
 *                        per layer (no per-pixel gather), validity per-tap
 *                        (clip range) not per-bit, patch words reused across
 *                        Cout. Wins at LARGE Cin (gather killed). OpenMP.
 *   bconv_dispatch     : picks per layer on Cin (measured x86 crossover ~36).
 *
 * Crossover is arch-dependent: it was measured on x86 with hardware POPCNT.
 * On ARM (no scalar popcount; __builtin_popcountll lowers to a NEON cnt
 * sequence) popcount is relatively costlier, which shifts the crossover and
 * favours the dense kernel / a NEON-intrinsic popcount. RE-MEASURE the
 * threshold on the target board; it is a single #define below.
 *
 * Padding matches PyTorch F.conv2d(padding=pad) on the real +-1 tensor:
 * padded taps contribute 0. So per output pixel,
 *     P = 2*popcount(XNOR(w,a) & valid) - popcount(valid),
 * valid = in-bounds taps (times Cin valid channel-bits each). For channel
 * packing an entire tap is in- or out-of-bounds, so validity reduces to a
 * (ky,kx) clip range plus a constant Cin-tail bit mask.
 *
 * K-index order (kidx layout): kidx = (ci*kh + ky)*kw + kx.
 *
 * Arrays contiguous row-major, batch size 1:
 *     a     : int8   [Cin, H, W]        values in {-1,+1}
 *     wsign : int8   [Cout, Cin, kh, kw] values in {-1,+1}
 *     P     : int32  [Cout, Hout, Wout]
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* x86 (hardware POPCNT) crossover, measured. Re-tune on ARM. */
#define BCONV_PACK_MIN_CIN 36

static inline int out_dim(int in, int k, int pad, int stride) {
    return (in + 2 * pad - k) / stride + 1;
}

void bconv_naive_int(const int8_t *a, const int8_t *wsign, int32_t *P,
                     int Cin, int H, int W, int Cout, int kh, int kw,
                     int pad, int stride) {
    int Hout = out_dim(H, kh, pad, stride);
    int Wout = out_dim(W, kw, pad, stride);
    for (int co = 0; co < Cout; ++co) {
        for (int oy = 0; oy < Hout; ++oy) {
            for (int ox = 0; ox < Wout; ++ox) {
                int32_t acc = 0;
                for (int ci = 0; ci < Cin; ++ci) {
                    for (int ky = 0; ky < kh; ++ky) {
                        int iy = oy * stride + ky - pad;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < kw; ++kx) {
                            int ix = ox * stride + kx - pad;
                            if (ix < 0 || ix >= W) continue;   /* padded -> 0 */
                            int8_t av = a[(ci * H + iy) * W + ix];
                            int8_t wv = wsign[((co * Cin + ci) * kh + ky) * kw + kx];
                            acc += (int32_t)wv * (int32_t)av;
                        }
                    }
                }
                P[(co * Hout + oy) * Wout + ox] = acc;
            }
        }
    }
}

/* pack sign bits (+1 -> 1, -1 -> 0) per output channel into 64-bit words,
 * in kidx order. packed has Cout * nwords words, nwords = ceil(K/64). */
void pack_weights(const int8_t *wsign, uint64_t *packed,
                  int Cout, int Cin, int kh, int kw) {
    int K = Cin * kh * kw;
    int nwords = (K + 63) / 64;
    memset(packed, 0, (size_t)Cout * nwords * sizeof(uint64_t));
    for (int co = 0; co < Cout; ++co) {
        uint64_t *dst = packed + (size_t)co * nwords;
        for (int ci = 0; ci < Cin; ++ci)
            for (int ky = 0; ky < kh; ++ky)
                for (int kx = 0; kx < kw; ++kx) {
                    int kidx = (ci * kh + ky) * kw + kx;
                    int8_t wv = wsign[((co * Cin + ci) * kh + ky) * kw + kx];
                    if (wv > 0) dst[kidx >> 6] |= (uint64_t)1 << (kidx & 63);
                }
    }
}

#define MAXWORDS 64   /* supports K up to 4096 (Cin*kh*kw) */

/* dense kidx-packed kernel; re-gathers the patch per output pixel. */
void bconv_xnor(const int8_t *a, const uint64_t *packed_w, int32_t *P,
                int Cin, int H, int W, int Cout, int kh, int kw,
                int pad, int stride) {
    int Hout = out_dim(H, kh, pad, stride);
    int Wout = out_dim(W, kw, pad, stride);
    int K = Cin * kh * kw;
    int nwords = (K + 63) / 64;

    #pragma omp parallel for schedule(static)
    for (int oy = 0; oy < Hout; ++oy) {
        uint64_t iv[MAXWORDS], mask[MAXWORDS];   /* per-thread patch bit vectors */
        for (int ox = 0; ox < Wout; ++ox) {
            memset(iv, 0, nwords * sizeof(uint64_t));
            memset(mask, 0, nwords * sizeof(uint64_t));
            for (int ci = 0; ci < Cin; ++ci) {
                for (int ky = 0; ky < kh; ++ky) {
                    int iy = oy * stride + ky - pad;
                    for (int kx = 0; kx < kw; ++kx) {
                        int ix = ox * stride + kx - pad;
                        int kidx = (ci * kh + ky) * kw + kx;
                        if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                            continue;                       /* padded: mask 0 */
                        mask[kidx >> 6] |= (uint64_t)1 << (kidx & 63);
                        if (a[(ci * H + iy) * W + ix] > 0)
                            iv[kidx >> 6] |= (uint64_t)1 << (kidx & 63);
                    }
                }
            }
            int valid_total = 0;
            for (int w = 0; w < nwords; ++w)
                valid_total += __builtin_popcountll(mask[w]);
            for (int co = 0; co < Cout; ++co) {
                const uint64_t *wv = packed_w + (size_t)co * nwords;
                int valid_equal = 0;
                for (int w = 0; w < nwords; ++w) {
                    uint64_t xnor = ~(wv[w] ^ iv[w]);       /* 1 where equal */
                    valid_equal += __builtin_popcountll(xnor & mask[w]);
                }
                P[(co * Hout + oy) * Wout + ox] = 2 * valid_equal - valid_total;
            }
        }
    }
}

/* channel-packed kernel; input packed once, no per-pixel gather. */
void bconv_xnor_packed(const int8_t *a, const uint64_t *packed_w, int32_t *P,
                       int Cin, int H, int W, int Cout, int kh, int kw,
                       int pad, int stride) {
    int Hout = out_dim(H, kh, pad, stride);
    int Wout = out_dim(W, kw, pad, stride);
    int Wc = (Cin + 63) / 64;                 /* channel words per tap */
    int K = Cin * kh * kw;
    int nwords = (K + 63) / 64;               /* old kidx-order word count */
    uint64_t tail = (Cin & 63) ? (((uint64_t)1 << (Cin & 63)) - 1) : ~(uint64_t)0;

    /* 1. pack input a[Cin,H,W] -> apack[H*W][Wc], channel-packed per pixel */
    uint64_t *apack = calloc((size_t)H * W * Wc, sizeof(uint64_t));
    if (!apack) { fprintf(stderr, "OOM apack\n"); exit(1); }
    for (int ci = 0; ci < Cin; ++ci) {
        int wo = ci >> 6; uint64_t bit = (uint64_t)1 << (ci & 63);
        const int8_t *ac = a + (size_t)ci * H * W;
        for (int p = 0; p < H * W; ++p)
            if (ac[p] > 0) apack[(size_t)p * Wc + wo] |= bit;
    }

    /* 2. repack weights kidx-order -> wpack[Cout][kh*kw][Wc], channel-per-tap.
     *    O(Cout*K), negligible vs the main loop. */
    uint64_t *wpack = calloc((size_t)Cout * kh * kw * Wc, sizeof(uint64_t));
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

    /* 3. main loop, parallel over output rows */
    #pragma omp parallel for schedule(static)
    for (int oy = 0; oy < Hout; ++oy) {
        for (int ox = 0; ox < Wout; ++ox) {
            int oyb = oy * stride - pad, oxb = ox * stride - pad;
            int ky0 = oyb < 0 ? -oyb : 0;
            int ky1 = (oyb + kh > H) ? H - oyb : kh;
            int kx0 = oxb < 0 ? -oxb : 0;
            int kx1 = (oxb + kw > W) ? W - oxb : kw;
            int valid_total = (ky1 - ky0) * (kx1 - kx0) * Cin;
            for (int co = 0; co < Cout; ++co) {
                const uint64_t *wco = wpack + (size_t)(co * kh) * kw * Wc;
                int equal = 0;
                for (int ky = ky0; ky < ky1; ++ky) {
                    int iy = oyb + ky;
                    for (int kx = kx0; kx < kx1; ++kx) {
                        int ix = oxb + kx;
                        const uint64_t *ap = apack + (size_t)(iy * W + ix) * Wc;
                        const uint64_t *wp = wco + (size_t)(ky * kw + kx) * Wc;
                        for (int wc = 0; wc < Wc; ++wc) {
                            uint64_t x = ~(ap[wc] ^ wp[wc]);
                            uint64_t m = (wc == Wc - 1) ? tail : ~(uint64_t)0;
                            equal += __builtin_popcountll(x & m);
                        }
                    }
                }
                P[((size_t)co * Hout + oy) * Wout + ox] = 2 * equal - valid_total;
            }
        }
    }
    free(apack); free(wpack);
}

/* pick the faster kernel for this layer's shape (same packed_w, same result) */
void bconv_dispatch(const int8_t *a, const uint64_t *packed_w, int32_t *P,
                    int Cin, int H, int W, int Cout, int kh, int kw,
                    int pad, int stride) {
    if (Cin >= BCONV_PACK_MIN_CIN)
        bconv_xnor_packed(a, packed_w, P, Cin, H, W, Cout, kh, kw, pad, stride);
    else
        bconv_xnor(a, packed_w, P, Cin, H, W, Cout, kh, kw, pad, stride);
}