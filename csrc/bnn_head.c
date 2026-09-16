/*
 * bnn_head.c - FP32 output head (outc): a plain real 1x1 convolution with
 * bias, the network's final layer. Input is the last +-1 activation
 * (dec1.act2 output), weights and bias are real FP32, output is real logits.
 *
 * A 1x1 conv is a per-pixel matrix-vector product:
 *     logits[o, y, x] = bias[o] + sum_c W[o, c] * a[c, y, x]
 *
 * The input is +-1 here, but the kernel treats it as general float (a 1x1
 * conv is a real matvec regardless of input domain), which also lets the
 * same code serve if the input were ever non-binary. Verified exact (0 diff)
 * against nn.Conv2d(1x1, bias=True) before this was written.
 *
 * Arrays, contiguous row-major, batch size 1:
 *     a      : float32 [Cin, H, W]        input activation (+-1 in practice)
 *     W      : float32 [Cout, Cin]        1x1 conv weights (kh=kw=1 squeezed)
 *     bias   : float32 [Cout]
 *     logits : float32 [Cout, H, W]
 */

void head_1x1(const float *a, const float *W, const float *bias,
              float *logits, int Cin, int H, int W_, int Cout) {
    int HW = H * W_;
    for (int o = 0; o < Cout; ++o) {
        const float *wrow = W + (long)o * Cin;
        float bo = bias[o];
        for (int p = 0; p < HW; ++p) {
            float acc = bo;
            for (int c = 0; c < Cin; ++c)
                acc += wrow[c] * a[(long)c * HW + p];
            logits[(long)o * HW + p] = acc;
        }
    }
}
