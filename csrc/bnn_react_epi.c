/* bnn_react_epi.c - NEON-vectorized epilogue candidates (track 2). Same
 * signature as react_epilogue_int/real in bnn_react_ops.c, so drop-in.
 * Elementwise, memory-bound: t = A*P + (B+move1) + res ; out = PReLU(t).
 * PReLU via vcgeq + vbslq (no branch). Scalar fallback computes the same
 * values (tol vs the current kernels: fma vs separate mul/add, ~1e-6). */
#include <stdint.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

void react_epilogue_int_v2(const int32_t *P, const float *res, float *out,
                           int C, int H, int W, const float *A, const float *B,
                           const float *move1, const float *prelu_w) {
    long HW = (long)H * W;
    for (int c = 0; c < C; ++c) {
        float Ac = A[c], Bm = B[c] + move1[c], wc = prelu_w[c];
        const int32_t *Pc = P + (long)c * HW;
        const float *rc = res + (long)c * HW;
        float *oc = out + (long)c * HW;
        long p = 0;
#ifdef __aarch64__
        float32x4_t vA = vdupq_n_f32(Ac), vB = vdupq_n_f32(Bm);
        float32x4_t vw = vdupq_n_f32(wc), vz = vdupq_n_f32(0.0f);
        for (; p + 4 <= HW; p += 4) {
            float32x4_t t = vaddq_f32(vmlaq_f32(vB, vA, vcvtq_f32_s32(vld1q_s32(Pc + p))),
                                      vld1q_f32(rc + p));
            vst1q_f32(oc + p, vbslq_f32(vcgeq_f32(t, vz), t, vmulq_f32(vw, t)));
        }
#endif
        for (; p < HW; ++p) { float t = Ac * (float)Pc[p] + Bm + rc[p]; oc[p] = t >= 0.0f ? t : wc * t; }
    }
}

void react_epilogue_real_v2(const float *cin, const float *res, float *out,
                            int C, int H, int W, const float *A, const float *B,
                            const float *move1, const float *prelu_w) {
    long HW = (long)H * W;
    for (int c = 0; c < C; ++c) {
        float Ac = A[c], Bm = B[c] + move1[c], wc = prelu_w[c];
        const float *xc = cin + (long)c * HW;
        const float *rc = res + (long)c * HW;
        float *oc = out + (long)c * HW;
        long p = 0;
#ifdef __aarch64__
        float32x4_t vA = vdupq_n_f32(Ac), vB = vdupq_n_f32(Bm);
        float32x4_t vw = vdupq_n_f32(wc), vz = vdupq_n_f32(0.0f);
        for (; p + 4 <= HW; p += 4) {
            float32x4_t t = vaddq_f32(vmlaq_f32(vB, vA, vld1q_f32(xc + p)), vld1q_f32(rc + p));
            vst1q_f32(oc + p, vbslq_f32(vcgeq_f32(t, vz), t, vmulq_f32(vw, t)));
        }
#endif
        for (; p < HW; ++p) { float t = Ac * xc[p] + Bm + rc[p]; oc[p] = t >= 0.0f ? t : wc * t; }
    }
}
