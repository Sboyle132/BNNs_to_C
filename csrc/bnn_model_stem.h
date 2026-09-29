/* bnn_model_stem.h - v2 (spectral-stem, A1b) blob layout + assembled forward.
 * PARALLEL to bnn_model.h; the v1 engine is untouched. Mirrors
 * python/bnn2c/blob.py's v2 path. Little-endian. Reuses the SAME kernels as
 * the v1 engine (linked from the shared kernel .c files) and the SAME
 * profiler (bnn_prof.h/.c): PROF(...) is a no-op without -DBNN_PROFILE, so
 * the plain bnn_infer_stem build is byte-for-byte unaffected by profiling
 * support existing in the source. Struct names are suffixed 'S' (ConvS/ActS)
 * so this header can coexist with bnn_model.h without symbol clashes if ever
 * included together. */
#ifndef BNN_MODEL_STEM_H
#define BNN_MODEL_STEM_H
#include <stdint.h>

/* ---- shared kernels (defined in the kernel .c files, identical to v1) ---- */
void bconv_dispatch(const int8_t *a, const uint64_t *packed_w, int32_t *P,
                int Cin, int H, int W, int Cout, int kh, int kw, int pad, int stride);
void conv_realin_naive(const float *x, const int8_t *wsign, const float *alpha,
                       float *out, int Cin, int H, int W, int Cout,
                       int kh, int kw, int pad, int stride);
void maxpool_P(const int32_t *P, int32_t *Pout, int C, int H, int W, int k, int stride);
void apply_fold_int(const int32_t *P, int8_t *out, int C, int H, int W,
                    const int32_t *type, const int32_t *t, const int32_t *lo, const int32_t *hi);
void apply_fold_real(const float *x, int8_t *out, int C, int H, int W,
                     const float *A, const float *B, const float *slope, const float *rsign);
void head_1x1(const float *a, const float *W, const float *bias,
              float *logits, int Cin, int H, int W_, int Cout);

/* ---- parsed blob records (identical shape to v1's Conv/Act, name-suffixed) ---- */
typedef struct {
    int tag;                 /* 1 = binary (packed), 2 = real (wsign+alpha) */
    int Cout, Cin, kh, kw, pad, stride, nwords;
    uint64_t *packed;
    int8_t *wsign;
    float *alpha;
} ConvS;

typedef struct {
    int kind;                /* 0 = integer-P fold, 1 = real fold */
    int C;
    int32_t *type, *t, *lo, *hi;
    float *A, *B, *slope, *rsign;
} ActS;

typedef struct {
    int version, base_channels, n_bands, n_classes, upsampler;
    int has_stem, n_stem_layers;   /* v2: has_stem==1, n_stem_layers==1 */
    /* stem: real-input binary-weight 1x1 conv + real fold */
    ConvS stem_conv;
    ActS  stem_act;
    /* body: enc1.conv1 is BINARY here (v2), unlike v1 -- uniform with enc2-4 */
    ConvS enc_conv1[4], enc_conv2[4];
    ActS  enc_act1[4], enc_act2_skip[4], enc_act2_down[4];
    ConvS bott_conv1, bott_conv2;
    ActS  bott_act1, bott_act2;
    ConvS dec_conv1[4], dec_conv2[4];
    ActS  dec_act1[4], dec_act2[4];
    int head_Cout, head_Cin;
    float *head_W, *head_bias;
} StemModel;

StemModel *bnn_stem_load(const char *path);
void bnn_stem_free(StemModel *m);
/* input: [n_bands, H, W] float; logits: [n_classes, H, W] float (caller allocs) */
void bnn_stem_forward(const StemModel *m, const float *input, int H, int W, float *logits);

#endif