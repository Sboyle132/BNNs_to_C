/* bnn_model_react.h - ReActNet (blob v3) engine: blob layout (reader side) +
 * assembled forward. Mirrors python/bnn2c/blob_react.py. Little-endian.
 * Structs carry an R suffix so this coexists with bnn_model.h / bnn_model_stem.h
 * (same pattern the stem engine used with S). */
#ifndef BNN_MODEL_REACT_H
#define BNN_MODEL_REACT_H
#include <stdint.h>

/* ---- existing kernels, REUSED unchanged ---- */
void bconv_dispatch(const int8_t *a, const uint64_t *packed_w, int32_t *P,
                    int Cin, int H, int W, int Cout, int kh, int kw, int pad, int stride);
void conv_realin_naive(const float *x, const int8_t *wsign, const float *alpha,
                       float *out, int Cin, int H, int W, int Cout,
                       int kh, int kw, int pad, int stride);
void maxpool_P(const int32_t *P, int32_t *Pout, int C, int H, int W, int k, int stride);
void head_1x1(const float *a, const float *W, const float *bias,
              float *logits, int Cin, int H, int W_, int Cout);

/* ---- new kernels (bnn_react_ops.c) ---- */
void react_epilogue_int(const int32_t *P, const float *res, float *out,
                        int C, int H, int W, const float *A, const float *B,
                        const float *move1, const float *prelu_w);
void react_epilogue_real(const float *cin, const float *res, float *out,
                         int C, int H, int W, const float *A, const float *B,
                         const float *move1, const float *prelu_w);
/* NEON-vectorized epilogues (bnn_react_epi.c); ~4x on A53, drop-in for the above */
void react_epilogue_int_v2(const int32_t *P, const float *res, float *out,
                           int C, int H, int W, const float *A, const float *B,
                           const float *move1, const float *prelu_w);
void react_epilogue_real_v2(const float *cin, const float *res, float *out,
                            int C, int H, int W, const float *A, const float *B,
                            const float *move1, const float *prelu_w);
void react_tap(const float *x, int8_t *out, int C, int H, int W, const float *rsign);
void maxpool_real(const float *x, float *out, int C, int H, int W, int k, int stride);
void react_proj_1x1(const float *x, const float *Wm, float *out,
                    int Cin, int H, int W, int Cout);
/* register-tiled binary conv (bnn_bconv_neon.c); 3x3/pad1/stride1 fast path */
void bconv_blk_b4p4(const int8_t *a, const uint64_t *packed_w, int32_t *P,
                    int Cin, int H, int W, int Cout, int kh, int kw, int pad, int stride);
/* register-blocked real-input kernels (bnn_react_blk.c), need Cout % 8 == 0 */
void react_proj_blk(const float *x, const float *Wm, float *out,
                    int Cin, int H, int W, int Cout);
void react_conv3_blk(const float *x, const int8_t *wsign, const float *alpha,
                     float *out, int Cin, int H, int W, int Cout);

/* ---- parsed blob records ---- */
typedef struct {
    int tag;                 /* 1 = binary (packed), 2 = real (wsign+alpha) */
    int Cout, Cin, kh, kw, pad, stride, nwords;
    uint64_t *packed;
    int8_t *wsign;
    float *alpha;
} ConvR;

typedef struct { int kind, C; float *A, *B, *move1, *prelu_w; } EpiR;  /* kind 0=int 1=real */
typedef struct { int C; float *rsign; } TapR;                          /* C==0: absent (enc1) */
typedef struct { int Cout, Cin; float *W; } ProjR;                     /* real 1x1, no bias */

typedef struct {
    int is_down;             /* 1: DownBlock (fork), 0: PlainBlock */
    int input_real;          /* enc1 only: conv1 consumes raw spectra, no tap1 */
    TapR  tap1, tap2;
    ConvR conv1, conv2;
    ProjR proj1;
    EpiR  epi1;
    EpiR  epi_a;             /* down: skip epilogue ; plain: epi2 */
    EpiR  epi_b;             /* down: pooled-down epilogue ; plain: unused */
} BlockR;

typedef struct {
    int version, base_channels, n_bands, n_classes, upsampler;
    BlockR enc[4], bott, dec[4];     /* dec[0..3] == dec4..dec1 */
    int head_Cout, head_Cin;
    float *head_W, *head_bias;
} ModelR;

ModelR *bnn_load_react(const char *path);
void bnn_free_react(ModelR *m);
/* input: [n_bands,H,W] float; logits: [n_classes,H,W] float (caller allocs) */
void bnn_forward_react(const ModelR *m, const float *input, int H, int W, float *logits);

#endif
