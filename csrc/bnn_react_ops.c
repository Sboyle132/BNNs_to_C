/*
 * bnn_react_ops.c - the ONLY new kernels for the ReActNet (v3) engine.
 *
 * ReActNet keeps a REAL-valued highway between blocks. Per unit:
 *     res = shortcut(x)                 (real: proj 1x1, or identity highway)
 *     a   = sign(x + rsign_bias)        (the tap: binarise conv input, +-1)
 *     h   = BN(binconv(a)) + res        (REAL)
 *     h   = RPReLU(h)                   (REAL; RPReLU post-add, reactnet form)
 * The conv path (a -> P) is UNCHANGED and uses the existing bconv/conv_realin.
 * Only the real epilogue (BN affine + real add + RPReLU) and the tap are new.
 *
 * BN folds into a per-channel affine on the conv accumulator:
 *   binary conv:  conv_out = alpha_k * P_k ->
 *       BN(conv_out) = A_k*P_k + B_k,  A = alpha*gamma/sqrt(var+eps)
 *   real conv (enc1.conv1): conv_out already carries alpha ->
 *       BN(conv_out) = A_k*c_k + B_k,  A = gamma/sqrt(var+eps)
 *   B = beta - gamma*mean/sqrt(var+eps)   (both cases)
 * RPReLU: t = (BN + res) + move1_bias ; out = t>=0 ? t : prelu_w * t
 * (move2/zeta are dropped in this family, same as the brevitas dualskip models.)
 *
 * No sign() here: the result stays real and goes onto the highway. Binarisation
 * happens later, in react_tap, right before the next conv.
 *
 * Arrays contiguous row-major, batch 1:
 *   P    : int32   [C,H,W]   (epilogue_int)
 *   cin  : float32 [C,H,W]   (epilogue_real, = real conv output)
 *   res  : float32 [C,H,W]   (real shortcut map: proj output / identity highway)
 *   out,x: float32 [C,H,W]
 *   A,B,move1,prelu_w,rsign : float32 [C]
 */

#include <stdint.h>
#include <math.h>

/* epilogue on an INTEGER accumulator P (binary conv). A folds alpha. */
void react_epilogue_int(const int32_t *P, const float *res, float *out,
                        int C, int H, int W,
                        const float *A, const float *B,
                        const float *move1, const float *prelu_w) {
    long HW = (long)H * W;
    for (int c = 0; c < C; ++c) {
        float Ac = A[c], Bc = B[c], mc = move1[c], wc = prelu_w[c];
        const int32_t *Pc = P + (long)c * HW;
        const float *rc = res + (long)c * HW;
        float *oc = out + (long)c * HW;
        for (long p = 0; p < HW; ++p) {
            float t = Ac * (float)Pc[p] + Bc + rc[p] + mc;
            oc[p] = (t >= 0.0f) ? t : wc * t;
        }
    }
}

/* epilogue on a REAL conv output (enc1.conv1 only). A does NOT fold alpha
 * (already applied inside conv_realin). Otherwise identical to the int form. */
void react_epilogue_real(const float *cin, const float *res, float *out,
                         int C, int H, int W,
                         const float *A, const float *B,
                         const float *move1, const float *prelu_w) {
    long HW = (long)H * W;
    for (int c = 0; c < C; ++c) {
        float Ac = A[c], Bc = B[c], mc = move1[c], wc = prelu_w[c];
        const float *xc = cin + (long)c * HW;
        const float *rc = res + (long)c * HW;
        float *oc = out + (long)c * HW;
        for (long p = 0; p < HW; ++p) {
            float t = Ac * xc[p] + Bc + rc[p] + mc;
            oc[p] = (t >= 0.0f) ? t : wc * t;
        }
    }
}

/* binarising tap off the real highway: out = sign(x + rsign_bias) in {-1,+1}.
 * Convention matches the model (bit = pre_sign > 0) and bnn_fold.c: r>0 -> +1
 * else -1, so exact-zero maps to -1 exactly as np.sign()->0 then ->-1 does. */
void react_tap(const float *x, int8_t *out, int C, int H, int W,
               const float *rsign) {
    long HW = (long)H * W;
    for (int c = 0; c < C; ++c) {
        float rc = rsign[c];
        const float *xc = x + (long)c * HW;
        int8_t *oc = out + (long)c * HW;
        for (long p = 0; p < HW; ++p)
            oc[p] = (xc[p] + rc > 0.0f) ? 1 : -1;
    }
}

/* real 2x2 maxpool on the highway (identity-shortcut down branch). Mirrors
 * maxpool_P (integer) but on float; same floor output dims as nn.MaxPool2d. */
void maxpool_real(const float *x, float *out, int C, int H, int W,
                  int k, int stride) {
    int Ho = (H - k) / stride + 1;
    int Wo = (W - k) / stride + 1;
    for (int c = 0; c < C; ++c) {
        const float *xc = x + (long)c * H * W;
        float *oc = out + (long)c * Ho * Wo;
        for (int oy = 0; oy < Ho; ++oy)
            for (int ox = 0; ox < Wo; ++ox) {
                float m = -INFINITY;
                for (int ky = 0; ky < k; ++ky)
                    for (int kx = 0; kx < k; ++kx) {
                        float v = xc[(oy * stride + ky) * W + (ox * stride + kx)];
                        if (v > m) m = v;
                    }
                oc[oy * Wo + ox] = m;
            }
    }
}

/* real fp32 1x1 projection (the short1.proj shortcut, no bias). Same result
 * as head_1x1 with zero bias (float summation-order noise only), but tiled
 * over pixels so each Cin x TILE input slab stays in L2 and is reused across
 * all Cout, with a contiguous streaming axpy inner loop (auto-vectorises).
 * head_1x1's inner loop strides by H*W over channels, which is cache-hostile
 * at enc1's 120-channel full-resolution input.
 *   x : [Cin,H,W]  Wm : [Cout,Cin]  out : [Cout,H,W] */
#define PROJ_TILE 256
void react_proj_1x1(const float *x, const float *Wm, float *out,
                    int Cin, int H, int W, int Cout) {
    long HW = (long)H * W;
    for (long p0 = 0; p0 < HW; p0 += PROJ_TILE) {
        long n = HW - p0 < PROJ_TILE ? HW - p0 : PROJ_TILE;
        for (int co = 0; co < Cout; ++co) {
            float *o = out + (long)co * HW + p0;
            const float *wr = Wm + (long)co * Cin;
            for (long i = 0; i < n; ++i) o[i] = 0.0f;
            for (int ci = 0; ci < Cin; ++ci) {
                float w = wr[ci];
                const float *xr = x + (long)ci * HW + p0;
                for (long i = 0; i < n; ++i) o[i] += w * xr[i];
            }
        }
    }
}
