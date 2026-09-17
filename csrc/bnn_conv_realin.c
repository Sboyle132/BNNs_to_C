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
 * v2: layout-transposed for vectorization. Profiling showed this op alone
 * is 83.5% of total forward time (enc1.conv1, 512x512: 8333 of 9928 ms),
 * dwarfing all 17 binary convs combined (15.4%) -- confirmed the actual
 * bottleneck, not the XNOR-popcount path. Root cause (gcc -fopt-info-vec-all):
 * the original loop keeps Cin (the dominant axis, 120) as the OUTER of the
 * three innermost loops with stride H*W in x, so the reduction axis is never
 * contiguous and gcc's auto-vectorizer can't touch it, independent of the
 * per-tap branch/bounds-check cleanup tried first (both left 0 loops
 * vectorized; only the layout fix does).
 *
 * Fix: transpose x [Cin,H,W] -> [H,W,Cin] and wsign [Cout,Cin,kh,kw] -> float
 * [Cout,kh,kw,Cin] once per call (O(H*W*Cin) and O(Cout*kh*kw*Cin), both
 * negligible next to the O(Cout*H*W*Cin*kh*kw) main compute), so the Cin
 * reduction is a contiguous float dot product -- gcc vectorizes it to AVX
 * (32-byte vectors) with zero source-level intrinsics. Boundary handling
 * unchanged in spirit: taps are clipped to a valid (ky,kx) range per output
 * pixel instead of checked per-tap, same zero-pad semantics as before.
 *
 * Verified (quick, not exhaustive): max abs diff ~2e-6 vs the original
 * branch/bounds-check kernel across several odd/edge sizes (H,W in
 * {1x1, 5x5, 7x4, 3x7, 16x16}) -- float summation-order noise only, no
 * logic divergence. Real 512x512x120->8 benchmark: 8648ms -> 1205ms (7.2x).
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
#include <stdlib.h>
#include <stdio.h>

static inline int out_dim_r(int in, int k, int pad, int stride) {
    return (in + 2 * pad - k) / stride + 1;
}

void conv_realin_naive(const float *x, const int8_t *wsign, const float *alpha,
                       float *out, int Cin, int H, int W, int Cout,
                       int kh, int kw, int pad, int stride) {
    int Hout = out_dim_r(H, kh, pad, stride);
    int Wout = out_dim_r(W, kw, pad, stride);

    /* x: [Cin,H,W] -> xt: [H,W,Cin], contiguous over Cin (the reduction axis) */
    float *xt = malloc((size_t)H * W * Cin * sizeof(float));
    if (!xt) { fprintf(stderr, "OOM xt\n"); exit(1); }
    for (int ci = 0; ci < Cin; ++ci)
        for (int y = 0; y < H; ++y)
            for (int xx = 0; xx < W; ++xx)
                xt[(y * W + xx) * Cin + ci] = x[(ci * H + y) * W + xx];

    /* wsign: [Cout,Cin,kh,kw] int8 -> wf: [Cout,kh,kw,Cin] float */
    float *wf = malloc((size_t)Cout * kh * kw * Cin * sizeof(float));
    if (!wf) { fprintf(stderr, "OOM wf\n"); exit(1); }
    for (int co = 0; co < Cout; ++co)
        for (int ci = 0; ci < Cin; ++ci)
            for (int ky = 0; ky < kh; ++ky)
                for (int kx = 0; kx < kw; ++kx)
                    wf[((co * kh + ky) * kw + kx) * Cin + ci] =
                        (float)wsign[((co * Cin + ci) * kh + ky) * kw + kx];

    for (int co = 0; co < Cout; ++co) {
        for (int oy = 0; oy < Hout; ++oy) {
            int oyb = oy * stride - pad;
            int ky0 = (oyb < 0) ? -oyb : 0;
            int ky1 = (oyb + kh > H) ? (H - oyb) : kh;
            for (int ox = 0; ox < Wout; ++ox) {
                int oxb = ox * stride - pad;
                int kx0 = (oxb < 0) ? -oxb : 0;
                int kx1 = (oxb + kw > W) ? (W - oxb) : kw;
                float acc = 0.0f;
                for (int ky = ky0; ky < ky1; ++ky) {
                    int iy = oyb + ky;
                    for (int kx = kx0; kx < kx1; ++kx) {
                        int ix = oxb + kx;
                        const float *xv = xt + (iy * W + ix) * Cin;
                        const float *wv = wf + ((co * kh + ky) * kw + kx) * Cin;
                        float s = 0.0f;
                        for (int ci = 0; ci < Cin; ++ci) s += xv[ci] * wv[ci];
                        acc += s;
                    }
                }
                out[(co * Hout + oy) * Wout + ox] = alpha[co] * acc;
            }
        }
    }
    free(xt); free(wf);
}