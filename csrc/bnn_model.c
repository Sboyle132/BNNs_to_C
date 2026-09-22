/* bnn_model.c - blob loader + assembled forward (nearest-upsampler variant).
 * The forward mirrors python/verify/forward_driver.py::run_chain exactly.
 * Per-op timing is opt-in via bnn_prof.h: PROF(...) is a no-op without
 * -DBNN_PROFILE, so the verified numerical path is byte-for-byte unchanged. */
#include "bnn_model.h"
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

static void load_conv(FILE *f, Conv *c) {
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

static void load_act(FILE *f, Act *a) {
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

Model *bnn_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[4]; rd(magic, 4, f);
    if (memcmp(magic, MAGIC, 4) != 0) { fprintf(stderr, "bad magic\n"); exit(1); }
    Model *m = xmalloc(sizeof(Model));
    m->version = rd_i(f); m->base_channels = rd_i(f);
    m->n_bands = rd_i(f); m->n_classes = rd_i(f); m->upsampler = rd_i(f);
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
    /* confirm no trailing bytes */
    char extra; if (fread(&extra, 1, 1, f) != 0) { fprintf(stderr, "trailing bytes in blob\n"); exit(1); }
    fclose(f);
    return m;
}

/* ---- fold dispatch ---- */
static void act(const Act *a, const int32_t *P, const float *xreal,
                int8_t *out, int C, int H, int W) {
    if (a->kind == 0)
        apply_fold_int(P, out, C, H, W, a->type, a->t, a->lo, a->hi);
    else
        apply_fold_real(xreal, out, C, H, W, a->A, a->B, a->slope, a->rsign);
}

/* binary conv (+-1 in) then integer fold -> +-1 out. `tag` names the stage
 * for the profiler (e.g. "enc2.c1" -> rows "enc2.c1.conv" / "enc2.c1.act"). */
static int8_t *bin_conv_act(const Conv *c, const Act *a, const int8_t *in,
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

/* nearest 2x upsample of +-1 x, padded up to the skip's size (PyTorch
 * F.pad([dw/2, dw-dw/2, dh/2, dh-dh/2]) convention: pad before = d/2), then
 * concat [skip, up] on channels. Handles odd skip dims (up smaller than skip).
 * Pad value 0 matches F.pad default; those positions are overwritten only
 * where the upsample covers them. */
static int8_t *up_cat(const int8_t *skip, int sC, int sH, int sW,
                      const int8_t *x, int xC, int hH, int hW, int *outC) {
    int8_t *out = calloc((size_t)(sC + xC) * sH * sW, 1);  /* zero-filled */
    if (!out) { fprintf(stderr, "OOM up_cat\n"); exit(1); }
    memcpy(out, skip, (size_t)sC * sH * sW);
    int upH = 2 * hH, upW = 2 * hW;
    int dh = sH - upH, dw = sW - upW;       /* >=0 when skip larger (odd dims) */
    int oh = dh > 0 ? dh / 2 : 0;           /* pad-before offset, height */
    int ow = dw > 0 ? dw / 2 : 0;           /* pad-before offset, width */
    int copyH = upH < sH ? upH : sH;        /* valid upsampled extent */
    int copyW = upW < sW ? upW : sW;
    for (int c = 0; c < xC; ++c)
        for (int y = 0; y < copyH; ++y)
            for (int xx = 0; xx < copyW; ++xx)
                out[((size_t)(sC + c) * sH + (oh + y)) * sW + (ow + xx)] =
                    x[((size_t)c * hH + (y / 2)) * hW + (xx / 2)];
    *outC = sC + xC;
    return out;
}

void bnn_forward(const Model *m, const float *input, int H, int W, float *logits) {
    prof_reset();
    int8_t *skip[4]; int skipC[4], skipH[4], skipW[4];
    char lbl[24];

    /* ---- enc1 (real first conv) ---- */
    int8_t *x; int xC, xH, xW;
    {
        const Conv *c1 = &m->enc_conv1[0];
        float *co1 = xmalloc((size_t)c1->Cout * H * W * sizeof(float));
        PROF("enc1.conv1", "conv_real",
             (double)c1->Cout * H * W, (double)c1->Cin * c1->kh * c1->kw,
             conv_realin_naive(input, c1->wsign, c1->alpha, co1,
                               c1->Cin, H, W, c1->Cout, c1->kh, c1->kw, c1->pad, c1->stride));
        int8_t *a1 = xmalloc((size_t)c1->Cout * H * W);
        PROF("enc1.act1", "fold", (double)c1->Cout * H * W, 0.0,
             act(&m->enc_act1[0], NULL, co1, a1, c1->Cout, H, W));
        free(co1);
        const Conv *c2 = &m->enc_conv2[0];
        int32_t *P2 = xmalloc((size_t)c2->Cout * H * W * sizeof(int32_t));
        PROF("enc1.conv2", "bconv",
             (double)c2->Cout * H * W, (double)c2->Cin * c2->kh * c2->kw,
             bconv_dispatch(a1, c2->packed, P2, c2->Cin, H, W, c2->Cout, c2->kh, c2->kw, c2->pad, c2->stride));
        free(a1);
        skip[0] = xmalloc((size_t)c2->Cout * H * W);
        PROF("enc1.skip", "fold", (double)c2->Cout * H * W, 0.0,
             act(&m->enc_act2_skip[0], P2, NULL, skip[0], c2->Cout, H, W));
        skipC[0] = c2->Cout; skipH[0] = H; skipW[0] = W;
        int Hh = H / 2, Wh = W / 2;
        int32_t *Pd = xmalloc((size_t)c2->Cout * Hh * Wh * sizeof(int32_t));
        PROF("enc1.pool", "maxpool", (double)c2->Cout * Hh * Wh, 4.0,
             maxpool_P(P2, Pd, c2->Cout, H, W, 2, 2));
        free(P2);
        x = xmalloc((size_t)c2->Cout * Hh * Wh);
        PROF("enc1.down", "fold", (double)c2->Cout * Hh * Wh, 0.0,
             act(&m->enc_act2_down[0], Pd, NULL, x, c2->Cout, Hh, Wh));
        free(Pd);
        xC = c2->Cout; xH = Hh; xW = Wh;
    }
    /* ---- enc2..4 (binary) ---- */
    for (int i = 1; i < 4; ++i) {
        int8_t *h = bin_conv_act(&m->enc_conv1[i], &m->enc_act1[i], x, xH, xW,
                                 PLABEL(lbl, "enc%d.c1", i + 1));
        free(x);
        const Conv *c2 = &m->enc_conv2[i];
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
        xC = m->bott_conv2.Cout;  /* xH,xW unchanged */
    }
    /* ---- decoder dec4..dec1 ---- */
    for (int i = 0; i < 4; ++i) {
        int si = 3 - i;                 /* dec4<-s4(enc3 idx3)... skip[3-i] */
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
    /* ---- head ---- */
    /* head takes float input; the last activation is +-1 int8, widen it */
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

static void free_conv(Conv *c) { free(c->packed); free(c->wsign); free(c->alpha); }
static void free_act(Act *a) { free(a->type); free(a->t); free(a->lo); free(a->hi);
                               free(a->A); free(a->B); free(a->slope); free(a->rsign); }
void bnn_free(Model *m) {
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