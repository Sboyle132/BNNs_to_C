/*
 * bnn_maxpool.c - integer max-pool on the binary-conv accumulator P, for the
 * encoder down branches.
 *
 * In each DualSkipDownBlock the down branch is MaxPool2d(2) applied to
 * conv2_out (the real alpha*P) BEFORE BN and sign. Since alpha_k = mean(|w|)
 * > 0 is a positive per-channel constant,
 *
 *     max_window(alpha_k * P) = alpha_k * max_window(P)
 *
 * so max-pooling commutes with the alpha scale and reduces to a plain 2x2
 * max over the integer accumulator P, per channel. Verified to float
 * precision (max diff ~1e-9) against F.max_pool2d on alpha*P before this was
 * written. The pooled P is then fed to the down-branch threshold exactly
 * like the unpooled P feeds the skip-branch threshold (shared conv2, two
 * consumers).
 *
 * Default kernel=2, stride=2, no padding (PyTorch MaxPool2d(2)). Floor
 * division on the output size matches PyTorch's default ceil_mode=False.
 *
 * Arrays, contiguous row-major:
 *     P    : int32 [C, H, W]
 *     Pout : int32 [C, H/k, W/k]
 */

#include <stdint.h>

void maxpool_P(const int32_t *P, int32_t *Pout, int C, int H, int W,
               int k, int stride) {
    int Hout = (H - k) / stride + 1;
    int Wout = (W - k) / stride + 1;
    for (int c = 0; c < C; ++c) {
        for (int oy = 0; oy < Hout; ++oy) {
            for (int ox = 0; ox < Wout; ++ox) {
                int iy0 = oy * stride;
                int ix0 = ox * stride;
                int32_t m = INT32_MIN;
                for (int dy = 0; dy < k; ++dy) {
                    for (int dx = 0; dx < k; ++dx) {
                        int32_t v = P[((long)c * H + (iy0 + dy)) * W + (ix0 + dx)];
                        if (v > m) m = v;
                    }
                }
                Pout[((long)c * Hout + oy) * Wout + ox] = m;
            }
        }
    }
}
