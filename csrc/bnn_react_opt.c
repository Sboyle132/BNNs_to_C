/* bnn_react_opt.c - independent NEON candidates (head, maxpool, tap->bitpack
 * fusion). Each is a drop-in with a scalar fallback computing identical values.
 * Benchmarked side-by-side by bench_opt.c; merged individually if they win. */
#include <stdint.h>
#include <string.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

/* ---- head: fp32 1x1 (Cout small, e.g. 16->3). Current head_1x1 strides the
 * input by H*W in the inner loop (cache-hostile); this streams contiguously
 * (out[o,:] += W[o,c]*a[c,:]) and vectorizes over pixels. ---- */
void head_1x1_v2(const float *a, const float *Wm, const float *bias,
                 float *logits, int Cin, int H, int W_, int Cout) {
    long HW = (long)H * W_;
    for (int o = 0; o < Cout; ++o) {
        float *lo = logits + (long)o * HW; const float *wr = Wm + (long)o * Cin;
        long p = 0;
#ifdef __aarch64__
        float32x4_t vb = vdupq_n_f32(bias[o]);
        for (p = 0; p + 4 <= HW; p += 4) vst1q_f32(lo + p, vb);
#endif
        for (; p < HW; ++p) lo[p] = bias[o];
        for (int c = 0; c < Cin; ++c) {
            float w = wr[c]; const float *ac = a + (long)c * HW;
            p = 0;
#ifdef __aarch64__
            float32x4_t vw = vdupq_n_f32(w);
            for (; p + 4 <= HW; p += 4) vst1q_f32(lo + p, vfmaq_f32(vld1q_f32(lo + p), vw, vld1q_f32(ac + p)));
#endif
            for (; p < HW; ++p) lo[p] += w * ac[p];
        }
    }
}

/* ---- maxpool 2x2 stride2, real and int (deinterleave + vmax) ---- */
void maxpool_real_v2(const float *x, float *out, int C, int H, int W, int k, int stride) {
    if (k != 2 || stride != 2) return;
    int Ho = (H - 2) / 2 + 1, Wo = (W - 2) / 2 + 1;
    for (int c = 0; c < C; ++c) {
        const float *xc = x + (long)c * H * W; float *oc = out + (long)c * Ho * Wo;
        for (int oy = 0; oy < Ho; ++oy) {
            const float *r0 = xc + (long)(2*oy) * W, *r1 = xc + (long)(2*oy+1) * W;
            float *o = oc + (long)oy * Wo; int ox = 0;
#ifdef __aarch64__
            for (; ox + 4 <= Wo; ox += 4) {
                float32x4x2_t a = vld2q_f32(r0 + 2*ox), b = vld2q_f32(r1 + 2*ox);
                vst1q_f32(o + ox, vmaxq_f32(vmaxq_f32(a.val[0], a.val[1]), vmaxq_f32(b.val[0], b.val[1])));
            }
#endif
            for (; ox < Wo; ++ox) {
                float m = r0[2*ox]; if (r0[2*ox+1] > m) m = r0[2*ox+1];
                if (r1[2*ox] > m) m = r1[2*ox];
                if (r1[2*ox+1] > m) m = r1[2*ox+1];
                o[ox] = m;
            }
        }
    }
}
void maxpool_P_v2(const int32_t *x, int32_t *out, int C, int H, int W, int k, int stride) {
    if (k != 2 || stride != 2) return;
    int Ho = (H - 2) / 2 + 1, Wo = (W - 2) / 2 + 1;
    for (int c = 0; c < C; ++c) {
        const int32_t *xc = x + (long)c * H * W; int32_t *oc = out + (long)c * Ho * Wo;
        for (int oy = 0; oy < Ho; ++oy) {
            const int32_t *r0 = xc + (long)(2*oy) * W, *r1 = xc + (long)(2*oy+1) * W;
            int32_t *o = oc + (long)oy * Wo; int ox = 0;
#ifdef __aarch64__
            for (; ox + 4 <= Wo; ox += 4) {
                int32x4x2_t a = vld2q_s32(r0 + 2*ox), b = vld2q_s32(r1 + 2*ox);
                vst1q_s32(o + ox, vmaxq_s32(vmaxq_s32(a.val[0], a.val[1]), vmaxq_s32(b.val[0], b.val[1])));
            }
#endif
            for (; ox < Wo; ++ox) {
                int32_t m = r0[2*ox]; if (r0[2*ox+1] > m) m = r0[2*ox+1];
                if (r1[2*ox] > m) m = r1[2*ox];
                if (r1[2*ox+1] > m) m = r1[2*ox+1];
                o[ox] = m;
            }
        }
    }
}

/* ---- tap->bitpack fusion: sign(x+rsign) written straight to channel-packed
 * bits (apack[p*Wc + ci>>6], bit ci&63), replacing the int8 tap + pack_input.
 * Wc==1 (the dominant slow layers, Cin<=64) uses a NEON movemask bitpack
 * (vcgtq -> replicate mask to 64b -> AND bit -> OR); other Wc fall back to a
 * fused scalar scatter. Output apack is identical to pack_input(tap(x)). ---- */
void tap_pack(const float *x, uint64_t *apack, int C, int H, int W, const float *rsign) {
    int Wc = (C + 63) / 64; long HW = (long)H * W;
    memset(apack, 0, (size_t)HW * Wc * sizeof(uint64_t));
    for (int ci = 0; ci < C; ++ci) {
        int wo = ci >> 6; uint64_t bit = (uint64_t)1 << (ci & 63); float rc = rsign[ci];
        const float *xc = x + (long)ci * HW;
        long p = 0;
#ifdef __aarch64__
        if (Wc == 1) {
            float32x4_t vr = vdupq_n_f32(rc), vz = vdupq_n_f32(0.0f);
            uint64x2_t vbit = vdupq_n_u64(bit);
            for (; p + 4 <= HW; p += 4) {
                uint32x4_t m = vcgtq_f32(vaddq_f32(vld1q_f32(xc + p), vr), vz);
                uint64x2_t m01 = vreinterpretq_u64_u32(vzip1q_u32(m, m));   /* pix 0,1 masks -> 64b */
                uint64x2_t m23 = vreinterpretq_u64_u32(vzip2q_u32(m, m));   /* pix 2,3 */
                vst1q_u64(apack + p,     vorrq_u64(vld1q_u64(apack + p),     vandq_u64(m01, vbit)));
                vst1q_u64(apack + p + 2, vorrq_u64(vld1q_u64(apack + p + 2), vandq_u64(m23, vbit)));
            }
        }
#endif
        for (; p < HW; ++p) if (xc[p] + rc > 0.0f) apack[(size_t)p * Wc + wo] |= bit;
    }
}
