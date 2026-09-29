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

/* conv_realin_general - the original transposed path (v2). BYTE-FOR-BYTE THE
 * SAME BODY as before this split (diagnostic sub-instrumentation removed
 * after doing its job -- see history below); only the name changed, so k=3
 * callers (v1's enc1.conv1) see identical behaviour, still routed here via
 * the conv_realin_naive dispatcher below. Reports as one row under the
 * caller's own PROF span (e.g. "enc1.conv1"), same as every other op.
 *
 * History: a since-removed per-stage PROF split (transpose_x/transpose_w/
 * compute) established that this transpose -- fixed at O(H*W*Cin) and
 * O(Cout*kh*kw*Cin), independent of kh*kw -- is ~12% of this function's own
 * time at k=3 (negligible, as the v2 docstring below claims) but ~37% at
 * k=1, which is why conv_realin_1x1 exists below for that shape and needs no
 * transpose at all. */
void conv_realin_general(const float *x, const int8_t *wsign, const float *alpha,
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

/* conv_realin_1x1 - fast path for kh=kw=1, pad=0, stride=1 (the spectral
 * stem's 1x1 reduction conv). NO transpose needed for either array:
 *   - wsign[co,ci,0,0] is already Cin-contiguous per co (kh=kw=1 collapses
 *     the last two dims to nothing), so w needs no reshuffle at all.
 *   - x[ci,:] is already one contiguous H*W block per ci in the ORIGINAL
 *     [Cin,H,W] layout -- so instead of transposing x to make a per-pixel
 *     dot product contiguous, the reduction is restructured as a sum of
 *     Cin rank-1 updates over the whole pixel plane at once:
 *         out[co,:] = alpha[co] * sum_ci w[co,ci] * x[ci,:]
 *     Each inner step (out[co,:] += w[co,ci]*x[ci,:]) is a plain contiguous
 *     streaming multiply-add over H*W elements -- the auto-vectorizer's
 *     easiest possible case, no gather/scatter, no transpose buffers.
 * Mathematically identical to conv_realin_general at this shape (same sum,
 * different order/associativity -- float summation-order noise only, same
 * tolerance class as the v1->v2 transpose optimisation's own ~2e-6 diff).
 * Verified against conv_realin_general as the oracle (bit-exact, 10
 * synthetic shapes incl. the real capture size). Reports as one row under
 * the caller's own PROF span (e.g. "stem.conv"), same convention as
 * conv_realin_general -- an earlier internal prof_add here double-counted
 * against that outer span (removed once it had nothing left to break down,
 * same reasoning as conv_realin_general's own removed sub-instrumentation
 * above). */
void conv_realin_1x1(const float *x, const int8_t *wsign, const float *alpha,
                     float *out, int Cin, int H, int W, int Cout) {
    long HW = (long)H * W;
    for (int co = 0; co < Cout; ++co) {
        float *orow = out + (long)co * HW;
        for (long p = 0; p < HW; ++p) orow[p] = 0.0f;
        const int8_t *wrow = wsign + (long)co * Cin;   /* Cin-contiguous, kh=kw=1 */
        for (int ci = 0; ci < Cin; ++ci) {
            float wv = (float)wrow[ci];
            const float *xrow = x + (long)ci * HW;      /* contiguous H*W block */
            for (long p = 0; p < HW; ++p) orow[p] += wv * xrow[p];
        }
        float a = alpha[co];
        for (long p = 0; p < HW; ++p) orow[p] *= a;
    }
}


/* conv_realin_naive - DISPATCHER, same name/signature every existing caller
 * (bnn_model.c, bnn_model_stem.c) already uses, so NEITHER file needs any
 * change: k=3 (v1's enc1.conv1) is routed to the untouched conv_realin_general
 * exactly as before this split; k=1,pad=0,stride=1 (the stem's 1x1) is
 * routed to the new fast path automatically. No flag, no config, nothing to
 * remember to set -- the shape alone decides, same pattern bconv_dispatch
 * already uses to pick the packed vs dense binary-conv kernel by Cin. */
void conv_realin_naive(const float *x, const int8_t *wsign, const float *alpha,
                       float *out, int Cin, int H, int W, int Cout,
                       int kh, int kw, int pad, int stride) {
    if (kh == 1 && kw == 1 && pad == 0 && stride == 1)
        conv_realin_1x1(x, wsign, alpha, out, Cin, H, W, Cout);
    else
        conv_realin_general(x, wsign, alpha, out, Cin, H, W, Cout, kh, kw, pad, stride);
}