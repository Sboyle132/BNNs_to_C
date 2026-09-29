/* bnn_model_stem.c - v2 blob loader + assembled forward for the spectral-stem
 * (A1b) variant. PARALLEL to bnn_model.c; the verified v1 engine is untouched.
 *
 * Difference from v1: a real-input 1x1 binary-weight STEM runs first and
 * binarises to +-1; the whole encoder (INCLUDING enc1) is then a UNIFORM
 * binary path routed through the same bin_conv_act helper (v1 gives enc1
 * bespoke inline code because it alone is real-input there; here the stem is
 * the odd one out, so enc1..enc4 share one code path). Bottleneck / decoder /
 * head are identical to v1.
 *
 * Per-op timing is opt-in via bnn_prof.h: PROF(...) is a no-op without
 * -DBNN_PROFILE, so the verified numerical path (already gated by
 * infer_capture.py --verify) is byte-for-byte unchanged whether or not this
 * file is built with profiling. Same instrument, same macros, same labelling
 * convention as bnn_model.c so stem/v1 profiles are directly comparable. */
#include "bnn_model_stem.h"
#include "bnn_prof.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "BNNC"
#define TAG_CONV_BIN 1
#define TAG_CONV_REAL 2
#define TAG_ACT 3
#define TAG_HEAD 4

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM %zu\n", n); exit(1); }
    return p;
}
static void rd(void *dst, size_t n, FILE *f) {
    if (fread(dst, 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(1); }
}
static int rd_i(FILE *f) { int v; rd(&v, 4, f); return v; }

static void load_conv(FILE *f, ConvS *c) {
    c->tag = rd_i(f);
    c->Cout = rd_i(f); c->Cin = rd_i(f); c->kh = rd_i(f); c->kw = rd_i(f);
    c->pad = rd_i(f); c->stride = rd_i(f);
    c->packed = NULL; c->wsign = NULL; c->alpha = NULL; c->nwords = 0;
    if (c->tag == TAG_CONV_BIN) {
        c->nwords = rd_i(f);
        size_t n = (size_t)c->Cout * c->nwords;
        c->packed = xmalloc(n * sizeof(uint64_t));
        rd(c->packed, n * sizeof(uint64_t), f);
    } else if (c->tag == TAG_CONV_REAL) {
        size_t nw = (size_t)c->Cout * c->Cin * c->kh * c->kw;
        c->wsign = xmalloc(nw);
        rd(c->wsign, nw, f);
        c->alpha = xmalloc((size_t)c->Cout * sizeof(float));
        rd(c->alpha, (size_t)c->Cout * sizeof(float), f);
    } else { fprintf(stderr, "bad conv tag %d\n", c->tag); exit(1); }
}

static void load_act(FILE *f, ActS *a) {
    int tag = rd_i(f);
    if (tag != TAG_ACT) { fprintf(stderr, "bad act tag %d\n", tag); exit(1); }
    a->kind = rd_i(f); a->C = rd_i(f);
    a->type = a->t = a->lo = a->hi = NULL;
    a->A = a->B = a->slope = a->rsign = NULL;
    size_t C = a->C;
    if (a->kind == 0) {
        a->type = xmalloc(C * 4); rd(a->type, C * 4, f);
        a->t = xmalloc(C * 4);    rd(a->t, C * 4, f);
        a->lo = xmalloc(C * 4);   rd(a->lo, C * 4, f);
        a->hi = xmalloc(C * 4);   rd(a->hi, C * 4, f);
    } else {
        a->A = xmalloc(C * 4);     rd(a->A, C * 4, f);
        a->B = xmalloc(C * 4);     rd(a->B, C * 4, f);
        a->slope = xmalloc(C * 4); rd(a->slope, C * 4, f);
        a->rsign = xmalloc(C * 4); rd(a->rsign, C * 4, f);
    }
}

StemModel *bnn_stem_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[4]; rd(magic, 4, f);
    if (memcmp(magic, MAGIC, 4) != 0) { fprintf(stderr, "bad magic\n"); exit(1); }
    StemModel *m = xmalloc(sizeof(StemModel));
    m->version = rd_i(f);
    if (m->version != 2) { fprintf(stderr, "bnn_stem: expected v2 blob, got v%d "
                                   "(use the v1 engine for non-stem models)\n", m->version); exit(1); }
    m->base_channels = rd_i(f);
    m->n_bands = rd_i(f); m->n_classes = rd_i(f); m->upsampler = rd_i(f);
    m->has_stem = rd_i(f); m->n_stem_layers = rd_i(f);
    if (m->has_stem != 1 || m->n_stem_layers != 1) {
        fprintf(stderr, "bnn_stem: only has_stem=1, n_stem_layers=1 supported\n"); exit(1);
    }
    load_conv(f, &m->stem_conv);
    load_act(f, &m->stem_act);
    for (int i = 0; i < 4; ++i) {
        load_conv(f, &m->enc_conv1[i]); load_act(f, &m->enc_act1[i]);
        load_conv(f, &m->enc_conv2[i]); load_act(f, &m->enc_act2_skip[i]);
        load_act(f, &m->enc_act2_down[i]);
    }
    load_conv(f, &m->bott_conv1); load_act(f, &m->bott_act1);
    load_conv(f, &m->bott_conv2); load_act(f, &m->bott_act2);
    for (int i = 0; i < 4; ++i) {
        load_conv(f, &m->dec_conv1[i]); load_act(f, &m->dec_act1[i]);
        load_conv(f, &m->dec_conv2[i]); load_act(f, &m->dec_act2[i]);
    }
    int tag = rd_i(f);
    if (tag != TAG_HEAD) { fprintf(stderr, "bad head tag %d\n", tag); exit(1); }
    m->head_Cout = rd_i(f); m->head_Cin = rd_i(f);
    m->head_W = xmalloc((size_t)m->head_Cout * m->head_Cin * 4);
    rd(m->head_W, (size_t)m->head_Cout * m->head_Cin * 4, f);
    m->head_bias = xmalloc((size_t)m->head_Cout * 4);
    rd(m->head_bias, (size_t)m->head_Cout * 4, f);
    char extra; if (fread(&extra, 1, 1, f) != 0) { fprintf(stderr, "trailing bytes in blob\n"); exit(1); }
    fclose(f);
    return m;
}

/* ---- fold dispatch (identical to v1) ---- */
static void act(const ActS *a, const int32_t *P, const float *xreal,
                int8_t *out, int C, int H, int W) {
    if (a->kind == 0)
        apply_fold_int(P, out, C, H, W, a->type, a->t, a->lo, a->hi);
    else
        apply_fold_real(xreal, out, C, H, W, a->A, a->B, a->slope, a->rsign);
}

/* binary conv (+-1 in) then integer fold -> +-1 out. `tag` names the stage
 * for the profiler (e.g. "enc1.c1" -> rows "enc1.c1.conv" / "enc1.c1.act").
 * Identical shape to bnn_model.c's bin_conv_act; here it also covers enc1,
 * since enc1 is no longer the real-input special case. */
static int8_t *bin_conv_act(const ConvS *c, const ActS *a, const int8_t *in,
                            int H, int W, const char *tag) {
    char lb[48];
    (void)tag;  /* used only by PLABEL under -DBNN_PROFILE */
    int32_t *P = xmalloc((size_t)c->Cout * H * W * sizeof(int32_t));
    PROF(PLABEL(lb, "%s.conv", tag), "bconv",
         (double)c->Cout * H * W, (double)c->Cin * c->kh * c->kw,
         bconv_dispatch(in, c->packed, P, c->Cin, H, W, c->Cout, c->kh, c->kw, c->pad, c->stride));
    int8_t *out = xmalloc((size_t)c->Cout * H * W);
    PROF(PLABEL(lb, "%s.act", tag), "fold",
         (double)c->Cout * H * W, 0.0,
         act(a, P, NULL, out, c->Cout, H, W));
    free(P);
    return out;
}

/* nearest 2x upsample of +-1 x, padded to skip size, concat [skip, up]
 * (identical to v1 up_cat). */
static int8_t *up_cat(const int8_t *skip, int sC, int sH, int sW,
                      const int8_t *x, int xC, int hH, int hW, int *outC) {
    int8_t *out = calloc((size_t)(sC + xC) * sH * sW, 1);
    if (!out) { fprintf(stderr, "OOM up_cat\n"); exit(1); }
    memcpy(out, skip, (size_t)sC * sH * sW);
    int upH = 2 * hH, upW = 2 * hW;
    int dh = sH - upH, dw = sW - upW;
    int oh = dh > 0 ? dh / 2 : 0;
    int ow = dw > 0 ? dw / 2 : 0;
    int copyH = upH < sH ? upH : sH;
    int copyW = upW < sW ? upW : sW;
    for (int c = 0; c < xC; ++c)
        for (int y = 0; y < copyH; ++y)
            for (int xx = 0; xx < copyW; ++xx)
                out[((size_t)(sC + c) * sH + (oh + y)) * sW + (ow + xx)] =
                    x[((size_t)c * hH + (y / 2)) * hW + (xx / 2)];
    *outC = sC + xC;
    return out;
}

void bnn_stem_forward(const StemModel *m, const float *input, int H, int W, float *logits) {
    prof_reset();
    int8_t *skip[4]; int skipC[4], skipH[4], skipW[4];
    char lbl[24];

    /* ---- STEM: real-input 1x1 binary conv -> real fold -> +-1 [stemC,H,W] ---- */
    int8_t *x; int xC, xH, xW;
    {
        const ConvS *sc = &m->stem_conv;
        float *co = xmalloc((size_t)sc->Cout * H * W * sizeof(float));
        PROF("stem.conv", "conv_real",
             (double)sc->Cout * H * W, (double)sc->Cin * sc->kh * sc->kw,
             conv_realin_naive(input, sc->wsign, sc->alpha, co,
                               sc->Cin, H, W, sc->Cout, sc->kh, sc->kw, sc->pad, sc->stride));
        x = xmalloc((size_t)sc->Cout * H * W);
        PROF("stem.act", "fold", (double)sc->Cout * H * W, 0.0,
             act(&m->stem_act, NULL, co, x, sc->Cout, H, W));
        free(co);
        xC = sc->Cout; xH = H; xW = W;
    }

    /* ---- encoder enc1..enc4: UNIFORM binary (enc1 is XNOR here) ---- */
    for (int i = 0; i < 4; ++i) {
        int8_t *h = bin_conv_act(&m->enc_conv1[i], &m->enc_act1[i], x, xH, xW,
                                 PLABEL(lbl, "enc%d.c1", i + 1));
        free(x);
        const ConvS *c2 = &m->enc_conv2[i];
        int32_t *P2 = xmalloc((size_t)c2->Cout * xH * xW * sizeof(int32_t));
        PROF(PLABEL(lbl, "enc%d.c2", i + 1), "bconv",
             (double)c2->Cout * xH * xW, (double)c2->Cin * c2->kh * c2->kw,
             bconv_dispatch(h, c2->packed, P2, c2->Cin, xH, xW, c2->Cout, c2->kh, c2->kw, c2->pad, c2->stride));
        free(h);
        skip[i] = xmalloc((size_t)c2->Cout * xH * xW);
        PROF(PLABEL(lbl, "enc%d.skip", i + 1), "fold",
             (double)c2->Cout * xH * xW, 0.0,
             act(&m->enc_act2_skip[i], P2, NULL, skip[i], c2->Cout, xH, xW));
        skipC[i] = c2->Cout; skipH[i] = xH; skipW[i] = xW;
        int Hh = xH / 2, Wh = xW / 2;
        int32_t *Pd = xmalloc((size_t)c2->Cout * Hh * Wh * sizeof(int32_t));
        PROF(PLABEL(lbl, "enc%d.pool", i + 1), "maxpool",
             (double)c2->Cout * Hh * Wh, 4.0,
             maxpool_P(P2, Pd, c2->Cout, xH, xW, 2, 2));
        free(P2);
        x = xmalloc((size_t)c2->Cout * Hh * Wh);
        PROF(PLABEL(lbl, "enc%d.down", i + 1), "fold",
             (double)c2->Cout * Hh * Wh, 0.0,
             act(&m->enc_act2_down[i], Pd, NULL, x, c2->Cout, Hh, Wh));
        free(Pd);
        xC = c2->Cout; xH = Hh; xW = Wh;
    }

    /* ---- bottleneck ---- */
    {
        int8_t *h = bin_conv_act(&m->bott_conv1, &m->bott_act1, x, xH, xW, "bott.c1"); free(x);
        x = bin_conv_act(&m->bott_conv2, &m->bott_act2, h, xH, xW, "bott.c2"); free(h);
        xC = m->bott_conv2.Cout;
    }

    /* ---- decoder dec4..dec1 ---- */
    for (int i = 0; i < 4; ++i) {
        int si = 3 - i;
        int H2 = skipH[si], W2 = skipW[si];
        int catC;
        int8_t *cat;
        PROF(PLABEL(lbl, "dec%d.upcat", 4 - i), "upcat",
             (double)(skipC[si] + xC) * H2 * W2, 0.0,
             cat = up_cat(skip[si], skipC[si], H2, W2, x, xC, xH, xW, &catC));
        free(x); free(skip[si]);
        int8_t *h = bin_conv_act(&m->dec_conv1[i], &m->dec_act1[i], cat, H2, W2,
                                 PLABEL(lbl, "dec%d.c1", 4 - i)); free(cat);
        x = bin_conv_act(&m->dec_conv2[i], &m->dec_act2[i], h, H2, W2,
                         PLABEL(lbl, "dec%d.c2", 4 - i)); free(h);
        xC = m->dec_conv2[i].Cout; xH = H2; xW = W2;
    }

    /* ---- head (float input; widen +-1 int8) ---- */
    {
        long n = (long)xC * xH * xW;
        float *xf = xmalloc(n * sizeof(float));
        PROF("head.widen", "widen", (double)n, 0.0,
             { for (long j = 0; j < n; ++j) xf[j] = (float)x[j]; });
        PROF("head", "head",
             (double)m->head_Cout * xH * xW, (double)m->head_Cin,
             head_1x1(xf, m->head_W, m->head_bias, logits, m->head_Cin, xH, xW, m->head_Cout));
        free(xf);
    }
    free(x);
}

static void free_conv(ConvS *c) { free(c->packed); free(c->wsign); free(c->alpha); }
static void free_act(ActS *a) { free(a->type); free(a->t); free(a->lo); free(a->hi);
                                free(a->A); free(a->B); free(a->slope); free(a->rsign); }
void bnn_stem_free(StemModel *m) {
    free_conv(&m->stem_conv); free_act(&m->stem_act);
    for (int i = 0; i < 4; ++i) {
        free_conv(&m->enc_conv1[i]); free_conv(&m->enc_conv2[i]);
        free_act(&m->enc_act1[i]); free_act(&m->enc_act2_skip[i]); free_act(&m->enc_act2_down[i]);
        free_conv(&m->dec_conv1[i]); free_conv(&m->dec_conv2[i]);
        free_act(&m->dec_act1[i]); free_act(&m->dec_act2[i]);
    }
    free_conv(&m->bott_conv1); free_conv(&m->bott_conv2);
    free_act(&m->bott_act1); free_act(&m->bott_act2);
    free(m->head_W); free(m->head_bias); free(m);
}