/*
 * bnn_conv_realin.c - convolution kernel for binary-weight, REAL-input convs:
 * enc1.conv1 (raw normalised float spectra) and dec{4,3,2,1}.conv1 (mixed
 * real-upsampled + binary-skip concat input, treated uniformly as real here).
 *
 * The model computes: conv_out = alpha_k * conv2d(x, sign(w))_k   (zero pad)
 * i.e. the SAME alpha-scaled sign-weight convention as the binary kernel,
 * but the activation x is float, not +-1, so this is a real multiply-
 * accumulate, not XNOR-popcount. Confirmed exactly (float precision) against
 * HardBinaryConv's own forward on real input before this was written.
 *
 * Padding matches PyTorch F.conv2d(padding=pad): out-of-bounds taps
 * contribute 0. No separate validity mask needed here (unlike the binary
 * XNOR kernel) because multiplying by a missing tap is already 0 in real
 * arithmetic; the loop just skips out-of-bounds indices, same as the naive
 * binary reference.
 *
 * Output is alpha_k * accumulator, i.e. this returns the conv_out directly
 * (already includes alpha), since there is no integer P to recover here,
 * unlike the pure-binary path -- the accumulator itself is real-valued and
 * that IS the model's floating output.
 *
 *     acc[co,y,x] = sum over (ci,ky,kx) of sign(w[co,ci,ky,kx]) * x[ci,y+ky-pad,x+kx-pad]
 *     out[co,y,x] = alpha[co] * acc[co,y,x]
 *
 * Arrays, contiguous row-major, batch size 1:
 *     x     : float32 [Cin, H, W]
 *     wsign : int8    [Cout, Cin, kh, kw]   values in {-1,+1}
 *     alpha : float32 [Cout]
 *     out   : float32 [Cout, Hout, Wout]
 */

#include <stdint.h>

static inline int out_dim_r(int in, int k, int pad, int stride) {
    return (in + 2 * pad - k) / stride + 1;
}

void conv_realin_naive(const float *x, const int8_t *wsign, const float *alpha,
                       float *out, int Cin, int H, int W, int Cout,
                       int kh, int kw, int pad, int stride) {
    int Hout = out_dim_r(H, kh, pad, stride);
    int Wout = out_dim_r(W, kw, pad, stride);
    for (int co = 0; co < Cout; ++co) {
        for (int oy = 0; oy < Hout; ++oy) {
            for (int ox = 0; ox < Wout; ++ox) {
                float acc = 0.0f;
                for (int ci = 0; ci < Cin; ++ci) {
                    for (int ky = 0; ky < kh; ++ky) {
                        int iy = oy * stride + ky - pad;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < kw; ++kx) {
                            int ix = ox * stride + kx - pad;
                            if (ix < 0 || ix >= W) continue;
                            float xv = x[(ci * H + iy) * W + ix];
                            int8_t wv = wsign[((co * Cin + ci) * kh + ky) * kw + kx];
                            acc += (wv > 0 ? xv : -xv);
                        }
                    }
                }
                out[(co * Hout + oy) * Wout + ox] = alpha[co] * acc;
            }
        }
    }
}
