/*
 * bnn_fold.c - activation fold kernels: turn a conv accumulator into the
 * next layer's +-1 activation.
 *
 * Two forms, matching the two chain types found by derive_thresholds.py:
 *
 * apply_fold_int: integer-P chains (all activations except enc1.act1). The
 *   per-channel rule was precomputed by folding alpha+BN+RPReLU+RSign into a
 *   decision on the integer accumulator P. No floats in this path.
 *     rule type per channel:
 *       0 const : out = (t>0 ? +1 : -1)                everywhere
 *       1 ge    : out = (P >= t)      ? +1 : -1
 *       2 lt    : out = (P <  t)      ? +1 : -1
 *       3 band_in : out = (lo<=P<hi)  ? +1 : -1
 *       4 band_out: out = (P<lo || P>=hi) ? +1 : -1
 *
 * apply_fold_real: real-accumulator chains (enc1.act1, whose conv input is
 *   the raw float spectra). Applies BN then RPReLU then RSign then sign,
 *   per channel, on the real conv output x:
 *       z = A*x + B ; z = (z>=0 ? z : slope*z) ; r = z + rsign ; out = sign(r)
 *   with A = gamma/sqrt(var+eps), B = beta - A*mean + move1_bias.
 *
 * Output is int8 in {-1,+1}. sign convention: r>0 -> +1 else -1 (matches the
 * model's bit = (pre_sign > 0); exact-zero pre-activations do not occur in
 * this model, confirmed by domain analysis).
 *
 * Arrays contiguous row-major:
 *   P    : int32   [C,H,W]     (int fold)
 *   x    : float32 [C,H,W]     (real fold)
 *   out  : int8    [C,H,W]
 *   per-channel rule/param arrays length C.
 */

#include <stdint.h>
#include <math.h>

void apply_fold_int(const int32_t *P, int8_t *out, int C, int H, int W,
                    const int32_t *type, const int32_t *t,
                    const int32_t *lo, const int32_t *hi) {
    int HW = H * W;
    for (int c = 0; c < C; ++c) {
        int ty = type[c];
        int tc = t[c], lc = lo[c], hc = hi[c];
        const int32_t *Pc = P + (long)c * HW;
        int8_t *oc = out + (long)c * HW;
        for (int p = 0; p < HW; ++p) {
            int v = Pc[p];
            int b;
            switch (ty) {
                case 0: b = (tc > 0); break;
                case 1: b = (v >= tc); break;
                case 2: b = (v < tc); break;
                case 3: b = (v >= lc && v < hc); break;
                case 4: b = (v < lc || v >= hc); break;
                default: b = 0; break;
            }
            oc[p] = b ? 1 : -1;
        }
    }
}

void apply_fold_real(const float *x, int8_t *out, int C, int H, int W,
                     const float *A, const float *B,
                     const float *slope, const float *rsign) {
    int HW = H * W;
    for (int c = 0; c < C; ++c) {
        float Ac = A[c], Bc = B[c], sc = slope[c], rc = rsign[c];
        const float *xc = x + (long)c * HW;
        int8_t *oc = out + (long)c * HW;
        for (int p = 0; p < HW; ++p) {
            float z = Ac * xc[p] + Bc;
            if (z < 0.0f) z = sc * z;
            float r = z + rc;
            oc[p] = (r > 0.0f) ? 1 : -1;
        }
    }
}
