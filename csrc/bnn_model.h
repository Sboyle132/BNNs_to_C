/* bnn_model.h - blob layout (reader side) + assembled forward.
 * Mirrors python/bnn2c/blob.py. Little-endian. */
#ifndef BNN_MODEL_H
#define BNN_MODEL_H
#include <stdint.h>

/* ---- kernels (defined in the other .c files) ---- */
void bconv_xnor(const int8_t *a, const uint64_t *packed_w, int32_t *P,
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

/* ---- parsed blob records ---- */
typedef struct {
    int tag;                 /* 1 = binary (packed), 2 = real (wsign+alpha) */
    int Cout, Cin, kh, kw, pad, stride, nwords;
    uint64_t *packed;        /* binary conv */
    int8_t *wsign;           /* real conv */
    float *alpha;            /* real conv */
} Conv;

typedef struct {
    int kind;                /* 0 = integer-P fold, 1 = real fold */
    int C;
    int32_t *type, *t, *lo, *hi;     /* int fold */
    float *A, *B, *slope, *rsign;    /* real fold */
} Act;

typedef struct {
    int version, base_channels, n_bands, n_classes, upsampler;
    /* encoder stages 0..3: conv1, act1, conv2, act2_skip, act2_down */
    Conv enc_conv1[4], enc_conv2[4];
    Act  enc_act1[4], enc_act2_skip[4], enc_act2_down[4];
    /* bottleneck */
    Conv bott_conv1, bott_conv2;
    Act  bott_act1, bott_act2;
    /* decoder stages 0..3 == dec4..dec1: conv1, act1, conv2, act2 */
    Conv dec_conv1[4], dec_conv2[4];
    Act  dec_act1[4], dec_act2[4];
    /* head */
    int head_Cout, head_Cin;
    float *head_W, *head_bias;
} Model;

Model *bnn_load(const char *path);
void bnn_free(Model *m);
/* input: [n_bands, H, W] float; logits: [n_classes, H, W] float (caller allocs) */
void bnn_forward(const Model *m, const float *input, int H, int W, float *logits);

#endif
