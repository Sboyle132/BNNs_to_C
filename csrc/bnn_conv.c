/*
 * bnn_conv.c - binary convolution kernels producing the integer accumulator
 * P for a W1A1 conv (activations in {-1,+1}, weights sign(w) in {-1,+1}).
 *
 * Two implementations, identical result, for A/B debugging:
 *   bconv_naive_int : plain triple loop in +-1 arithmetic. Trivially correct,
 *                     establishes the P reference.
 *   bconv_xnor      : bitpacked XNOR + popcount, the LCE/daBNN/FINN kernel form.
 *
 * Padding matches PyTorch F.conv2d(padding=pad) applied to the real +-1
 * tensor: padded taps contribute 0 (NOT -1). So for a given output pixel,
 *
 *     P = sum over in-bounds taps of  (wsign * a)             (each +-1)
 *       = #(valid & equal) - #(valid & different)
 *       = 2*popcount(XNOR(w,a) & valid) - popcount(valid).
 *
 * For interior pixels popcount(valid) == K (= Cin*kh*kw); at borders it is
 * smaller. The bitpacked kernel therefore carries a per-pixel validity mask.
 *
 * K-index order (weights and gathered input use the SAME order):
 *     kidx = (ci*kh + ky)*kw + kx
 *
 * All arrays are contiguous row-major, batch size 1:
 *     a     : int8   [Cin, H, W]        values in {-1,0,+1}
 *     wsign : int8   [Cout, Cin, kh, kw] values in {-1,+1} (sign of weight)
 *     P     : int32  [Cout, Hout, Wout]
 */

#include <stdint.h>
#include <string.h>

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

#define MAXWORDS 64   /* supports K up to 4096 */

void bconv_xnor(const int8_t *a, const uint64_t *packed_w, int32_t *P,
                int Cin, int H, int W, int Cout, int kh, int kw,
                int pad, int stride) {
    int Hout = out_dim(H, kh, pad, stride);
    int Wout = out_dim(W, kw, pad, stride);
    int K = Cin * kh * kw;
    int nwords = (K + 63) / 64;

    uint64_t iv[MAXWORDS];      /* gathered input bits */
    uint64_t mask[MAXWORDS];    /* validity mask (1 = in-bounds tap) */

    for (int oy = 0; oy < Hout; ++oy) {
        for (int ox = 0; ox < Wout; ++ox) {
            memset(iv, 0, nwords * sizeof(uint64_t));
            memset(mask, 0, nwords * sizeof(uint64_t));
            /* gather this output pixel's patch into bit vectors */
            for (int ci = 0; ci < Cin; ++ci) {
                for (int ky = 0; ky < kh; ++ky) {
                    int iy = oy * stride + ky - pad;
                    for (int kx = 0; kx < kw; ++kx) {
                        int ix = ox * stride + kx - pad;
                        int kidx = (ci * kh + ky) * kw + kx;
                        if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                            continue;                       /* padded: mask 0 */
                        mask[kidx >> 6] |= (uint64_t)1 << (kidx & 63);
                        int8_t av = a[(ci * H + iy) * W + ix];
                        if (av > 0)
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
