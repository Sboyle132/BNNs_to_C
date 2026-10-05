/* bnn_model_react.c - blob v3 loader + assembled ReActNet forward (nearest
 * upsampler). The forward mirrors python/verify (run_chain_react spec in
 * verify_react.py) exactly. Per-op timing is opt-in via bnn_prof.h, same
 * convention as bnn_model.c: PROF(...) is a no-op without -DBNN_PROFILE.
 *
 * Dataflow (REAL highway; the only sign() is the tap right before each conv):
 *   unit1:  res = proj(x) ; a = tap1(x) ; P = bconv(a) ; h1 = epi1(P) + res
 *           (enc1: c = conv_realin(x) ; h1 = epi1_real(c) + res, no tap)
 *   unit2:  a = tap2(h1) ; P2 = bconv(a)            (conv2 computed ONCE)
 *     down: skip = epi_skip(P2, h1)
 *           down = epi_down(maxpool_P_v2(P2), maxpool_real_v2(h1))
 *     plain: h2 = epi2(P2, h1)
 * Epilogues write in place over the shortcut buffer where legal (elementwise,
 * same index in/out), so no extra highway-sized allocation per unit. */
#include "bnn_model_react.h"
#include "bnn_prof.h"
#include "bnn_react_arena.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "BNNC"
#define VERSION_REACT 3
#define TAG_CONV_BIN 1
#define TAG_CONV_REAL 2
#define TAG_HEAD 4
#define TAG_PROJ 5
#define TAG_EPI 6
#define TAG_TAP 7

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM %zu\n", n); exit(1); }
    return p;
}
static void rd(void *dst, size_t n, FILE *f) {
    if (fread(dst, 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(1); }
}
static int rd_i(FILE *f) { int v; rd(&v, 4, f); return v; }
static float *rd_f(FILE *f, size_t n) {
    float *p = xmalloc(n * sizeof(float)); rd(p, n * sizeof(float), f); return p;
}

static void load_conv(FILE *f, ConvR *c) {
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
        c->wsign = xmalloc(nw); rd(c->wsign, nw, f);
        c->alpha = rd_f(f, (size_t)c->Cout);
    } else { fprintf(stderr, "bad conv tag %d\n", c->tag); exit(1); }
}

static void load_proj(FILE *f, ProjR *p) {
    int tag = rd_i(f);
    if (tag != TAG_PROJ) { fprintf(stderr, "bad proj tag %d\n", tag); exit(1); }
    p->Cout = rd_i(f); p->Cin = rd_i(f);
    p->W = rd_f(f, (size_t)p->Cout * p->Cin);
}

static void load_epi(FILE *f, EpiR *e) {
    int tag = rd_i(f);
    if (tag != TAG_EPI) { fprintf(stderr, "bad epi tag %d\n", tag); exit(1); }
    e->kind = rd_i(f); e->C = rd_i(f);
    e->A = rd_f(f, e->C); e->B = rd_f(f, e->C);
    e->move1 = rd_f(f, e->C); e->prelu_w = rd_f(f, e->C);
}

static void load_tap(FILE *f, TapR *t) {
    int tag = rd_i(f);
    if (tag != TAG_TAP) { fprintf(stderr, "bad tap tag %d\n", tag); exit(1); }
    t->C = rd_i(f); t->rsign = rd_f(f, t->C);
}

static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "blob layout mismatch: %s\n", what); exit(1); }
}

static void load_block(FILE *f, BlockR *b, int is_down, int input_real) {
    memset(b, 0, sizeof(*b));
    b->is_down = is_down; b->input_real = input_real;
    if (!input_real) load_tap(f, &b->tap1);
    load_conv(f, &b->conv1);
    load_proj(f, &b->proj1);
    load_epi(f, &b->epi1);
    load_tap(f, &b->tap2);
    load_conv(f, &b->conv2);
    load_epi(f, &b->epi_a);
    if (is_down) load_epi(f, &b->epi_b);
    /* cheap structural cross-checks: catch a stale/wrong blob at load time */
    check(b->conv1.tag == (input_real ? TAG_CONV_REAL : TAG_CONV_BIN), "conv1 tag");
    check(b->conv2.tag == TAG_CONV_BIN, "conv2 tag");
    check(b->proj1.Cin == b->conv1.Cin && b->proj1.Cout == b->conv1.Cout, "proj1 dims");
    check(input_real || b->tap1.C == b->conv1.Cin, "tap1 channels");
    check(b->tap2.C == b->conv2.Cin, "tap2 channels");
    check(b->epi1.C == b->conv1.Cout && b->epi1.kind == (input_real ? 1 : 0), "epi1");
    check(b->epi_a.C == b->conv2.Cout && b->epi_a.kind == 0, "epi_a");
    check(!is_down || (b->epi_b.C == b->conv2.Cout && b->epi_b.kind == 0), "epi_b");
}

ModelR *bnn_load_react(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[4]; rd(magic, 4, f);
    if (memcmp(magic, MAGIC, 4) != 0) { fprintf(stderr, "bad magic\n"); exit(1); }
    ModelR *m = xmalloc(sizeof(ModelR));
    m->version = rd_i(f);
    if (m->version != VERSION_REACT) {
        fprintf(stderr, "blob version %d, this engine reads v%d (ReActNet)\n",
                m->version, VERSION_REACT);
        exit(1);
    }
    m->base_channels = rd_i(f); m->n_bands = rd_i(f);
    m->n_classes = rd_i(f); m->upsampler = rd_i(f);
    if (m->upsampler != 0) { fprintf(stderr, "only nearest upsampler supported\n"); exit(1); }
    for (int i = 0; i < 4; ++i) load_block(f, &m->enc[i], 1, i == 0);
    load_block(f, &m->bott, 0, 0);
    for (int i = 0; i < 4; ++i) load_block(f, &m->dec[i], 0, 0);
    int tag = rd_i(f);
    if (tag != TAG_HEAD) { fprintf(stderr, "bad head tag %d\n", tag); exit(1); }
    m->head_Cout = rd_i(f); m->head_Cin = rd_i(f);
    m->head_W = rd_f(f, (size_t)m->head_Cout * m->head_Cin);
    m->head_bias = rd_f(f, (size_t)m->head_Cout);
    char extra; if (fread(&extra, 1, 1, f) != 0) { fprintf(stderr, "trailing bytes in blob\n"); exit(1); }
    fclose(f);
    rs_enable(1);
    return m;
}

/* ---------------------------------------------------------------- ops ---- */
/* real-input kernels: register-blocked versions when the shape allows (all
 * layers of this model do), else the previous general kernels. */
static void proj_dispatch(const ProjR *p, const float *x, float *out, int H, int W) {
    if (p->Cout % 8 == 0)
        react_proj_blk(x, p->W, out, p->Cin, H, W, p->Cout);
    else
        react_proj_1x1(x, p->W, out, p->Cin, H, W, p->Cout);
}
static void bconv_react(const ConvR *c, const int8_t *a, int32_t *P, int H, int W) {
    if (c->kh == 3 && c->kw == 3 && c->pad == 1 && c->stride == 1)
        bconv_blk_b4p4(a, c->packed, P, c->Cin, H, W, c->Cout, 3, 3, 1, 1);
    else
        bconv_dispatch(a, c->packed, P, c->Cin, H, W, c->Cout, c->kh, c->kw, c->pad, c->stride);
}
static void conv_real_dispatch(const ConvR *c, const float *x, float *out, int H, int W) {
    if (c->kh == 3 && c->kw == 3 && c->pad == 1 && c->stride == 1 && c->Cout % 8 == 0)
        react_conv3_blk(x, c->wsign, c->alpha, out, c->Cin, H, W, c->Cout);
    else
        conv_realin_naive(x, c->wsign, c->alpha, out, c->Cin, H, W, c->Cout,
                          c->kh, c->kw, c->pad, c->stride);
}

/* real fp32 1x1 shortcut projection: new buffer [Cout,H,W] */
static float *proj_apply(const ProjR *p, const float *x, int H, int W, const char *tag) {
    char lb[48]; (void)tag; (void)lb;
    float *out = rs_alloc((size_t)p->Cout * H * W * sizeof(float));
    PROF(PLABEL(lb, "%s.proj", tag), "proj",
         (double)p->Cout * H * W, (double)p->Cin,
         proj_dispatch(p, x, out, H, W));
    return out;
}

/* binarising tap off the real highway -> +-1 int8 */
static int8_t *tap_apply(const TapR *t, const float *x, int H, int W, const char *tag) {
    char lb[48]; (void)tag; (void)lb;
    int8_t *a = rs_alloc((size_t)t->C * H * W);
    PROF(PLABEL(lb, "%s.tap", tag), "tap",
         (double)t->C * H * W, 0.0,
         react_tap(x, a, t->C, H, W, t->rsign));
    return a;
}

/* binary conv on a +-1 tap -> integer P */
static int32_t *bconv_apply(const ConvR *c, const int8_t *a, int H, int W, const char *tag) {
    char lb[48]; (void)tag; (void)lb;
    int32_t *P = rs_alloc((size_t)c->Cout * H * W * sizeof(int32_t));
    PROF(PLABEL(lb, "%s.conv", tag), "bconv",
         (double)c->Cout * H * W, (double)c->Cin * c->kh * c->kw,
         bconv_react(c, a, P, H, W));
    return P;
}

/* unit1 for a binary-input block: returns h1 (new buffer, REAL) */
static float *unit1_bin(const BlockR *b, const float *x, int H, int W, const char *tag) {
    char lb[48]; (void)lb;
    float *res = proj_apply(&b->proj1, x, H, W, tag);
    int8_t *a1 = tap_apply(&b->tap1, x, H, W, tag);
    int32_t *P1 = bconv_apply(&b->conv1, a1, H, W, tag);
    rs_free(a1);
    PROF(PLABEL(lb, "%s.epi", tag), "epi",
         (double)b->conv1.Cout * H * W, 0.0,
         react_epilogue_int_v2(P1, res, res, b->conv1.Cout, H, W,
                            b->epi1.A, b->epi1.B, b->epi1.move1, b->epi1.prelu_w));
    rs_free(P1);
    return res;
}

/* unit2 for a plain block: consumes h1, returns h2 (written in place) */
static float *unit2_plain(const BlockR *b, float *h1, int H, int W, const char *tag) {
    char lb[48]; (void)lb;
    int8_t *a2 = tap_apply(&b->tap2, h1, H, W, tag);
    int32_t *P2 = bconv_apply(&b->conv2, a2, H, W, tag);
    rs_free(a2);
    PROF(PLABEL(lb, "%s.epi", tag), "epi",
         (double)b->conv2.Cout * H * W, 0.0,
         react_epilogue_int_v2(P2, h1, h1, b->conv2.Cout, H, W,
                            b->epi_a.A, b->epi_a.B, b->epi_a.move1, b->epi_a.prelu_w));
    rs_free(P2);
    return h1;
}

/* DownBlock: x real [Cin,H,W] in; skip real [Cout,H,W], down real [Cout,H/2,W/2]
 * out. x is NOT freed here (enc1's x is the caller's input). */
static void down_block(const BlockR *b, const float *x, int H, int W,
                       float **skip_out, float **down_out, const char *tag) {
    char lb[48], t1[48]; (void)lb; (void)t1; (void)tag;
    int Co = b->conv2.Cout;
    float *h1;
    if (b->input_real) {
        /* enc1 unit1: raw real spectra, binary-weight conv + real fold, no tap */
        const ConvR *c1 = &b->conv1;
        float *res = proj_apply(&b->proj1, x, H, W, PLABEL(t1, "%s.u1", tag));
        float *co1 = rs_alloc((size_t)c1->Cout * H * W * sizeof(float));
        PROF(PLABEL(lb, "%s.u1.conv", tag), "conv_real",
             (double)c1->Cout * H * W, (double)c1->Cin * c1->kh * c1->kw,
             conv_real_dispatch(c1, x, co1, H, W));
        PROF(PLABEL(lb, "%s.u1.epi", tag), "epi",
             (double)c1->Cout * H * W, 0.0,
             react_epilogue_real_v2(co1, res, res, c1->Cout, H, W,
                                 b->epi1.A, b->epi1.B, b->epi1.move1, b->epi1.prelu_w));
        rs_free(co1);
        h1 = res;
    } else {
        h1 = unit1_bin(b, x, H, W, PLABEL(t1, "%s.u1", tag));
    }

    /* unit2: conv2 once, then fork */
    char t2b[48];
    const char *t2 = PLABEL(t2b, "%s.u2", tag);   /* NULL w/o -DBNN_PROFILE; unused then */
    int8_t *a2 = tap_apply(&b->tap2, h1, H, W, t2);
    int32_t *P2 = bconv_apply(&b->conv2, a2, H, W, t2);
    rs_free(a2);

    float *skip = rs_alloc((size_t)Co * H * W * sizeof(float));
    PROF(PLABEL(lb, "%s.skip", tag), "epi",
         (double)Co * H * W, 0.0,
         react_epilogue_int_v2(P2, h1, skip, Co, H, W,
                            b->epi_a.A, b->epi_a.B, b->epi_a.move1, b->epi_a.prelu_w));

    int Hh = H / 2, Wh = W / 2;
    int32_t *Pd = rs_alloc((size_t)Co * Hh * Wh * sizeof(int32_t));
    PROF(PLABEL(lb, "%s.pool", tag), "maxpool",
         (double)Co * Hh * Wh, 4.0,
         maxpool_P_v2(P2, Pd, Co, H, W, 2, 2));
    rs_free(P2);
    float *h1d = rs_alloc((size_t)Co * Hh * Wh * sizeof(float));
    PROF(PLABEL(lb, "%s.poolr", tag), "maxpool",
         (double)Co * Hh * Wh, 4.0,
         maxpool_real_v2(h1, h1d, Co, H, W, 2, 2));
    rs_free(h1);
    PROF(PLABEL(lb, "%s.down", tag), "epi",
         (double)Co * Hh * Wh, 0.0,
         react_epilogue_int_v2(Pd, h1d, h1d, Co, Hh, Wh,
                            b->epi_b.A, b->epi_b.B, b->epi_b.move1, b->epi_b.prelu_w));
    rs_free(Pd);
    *skip_out = skip; *down_out = h1d;
}

/* nearest 2x upsample of real x, padded up to the skip's size (PyTorch
 * F.pad([dw/2, dw-dw/2, dh/2, dh-dh/2]): pad before = d/2), then concat
 * [skip, up] on channels. Float twin of bnn_model.c's int8 up_cat, same
 * odd-dim handling; pad value 0 matches F.pad default. */
static float *up_cat_f(const float *skip, int sC, int sH, int sW,
                       const float *x, int xC, int hH, int hW, int *outC) {
    float *out = rs_calloc((size_t)(sC + xC) * sH * sW * sizeof(float));
    if (!out) { fprintf(stderr, "OOM up_cat\n"); exit(1); }
    memcpy(out, skip, (size_t)sC * sH * sW * sizeof(float));
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

void bnn_forward_react(const ModelR *m, const float *input, int H, int W, float *logits) {
    prof_reset();
    float *skip[4]; int skipC[4], skipH[4], skipW[4];
    char lbl[24];

    /* ---- encoder ---- */
    float *x; int xC, xH, xW;
    down_block(&m->enc[0], input, H, W, &skip[0], &x, "enc1");
    skipC[0] = m->enc[0].conv2.Cout; skipH[0] = H; skipW[0] = W;
    xC = skipC[0]; xH = H / 2; xW = W / 2;
    for (int i = 1; i < 4; ++i) {
        float *nx;
        down_block(&m->enc[i], x, xH, xW, &skip[i], &nx, PLABEL(lbl, "enc%d", i + 1));
        rs_free(x); x = nx;
        skipC[i] = m->enc[i].conv2.Cout; skipH[i] = xH; skipW[i] = xW;
        xC = skipC[i]; xH /= 2; xW /= 2;
    }
    /* ---- bottleneck ---- */
    {
        float *h1 = unit1_bin(&m->bott, x, xH, xW, "bott.u1"); rs_free(x);
        x = unit2_plain(&m->bott, h1, xH, xW, "bott.u2");
        xC = m->bott.conv2.Cout;
    }
    /* ---- decoder dec4..dec1 ---- */
    for (int i = 0; i < 4; ++i) {
        int si = 3 - i;
        int H2 = skipH[si], W2 = skipW[si];
        int catC;
        float *cat;
        PROF(PLABEL(lbl, "dec%d.upcat", 4 - i), "upcat",
             (double)(skipC[si] + xC) * H2 * W2, 0.0,
             cat = up_cat_f(skip[si], skipC[si], H2, W2, x, xC, xH, xW, &catC));
        rs_free(x); rs_free(skip[si]);
        char t1b[32], t2b[32];
        const char *t1 = PLABEL(t1b, "dec%d.u1", 4 - i);
        const char *t2 = PLABEL(t2b, "dec%d.u2", 4 - i);
        float *h1 = unit1_bin(&m->dec[i], cat, H2, W2, t1); rs_free(cat);
        x = unit2_plain(&m->dec[i], h1, H2, W2, t2);
        xC = m->dec[i].conv2.Cout; xH = H2; xW = W2;
    }
    /* ---- head: x is already real, no widen ---- */
    PROF("head", "head",
         (double)m->head_Cout * xH * xW, (double)m->head_Cin,
         head_1x1_v2(x, m->head_W, m->head_bias, logits, m->head_Cin, xH, xW, m->head_Cout));
    rs_free(x);
}

static void free_conv(ConvR *c) { free(c->packed); free(c->wsign); free(c->alpha); }
static void free_epi(EpiR *e) { free(e->A); free(e->B); free(e->move1); free(e->prelu_w); }
static void free_block(BlockR *b) {
    free(b->tap1.rsign); free(b->tap2.rsign);
    free_conv(&b->conv1); free_conv(&b->conv2);
    free(b->proj1.W);
    free_epi(&b->epi1); free_epi(&b->epi_a);
    if (b->is_down) free_epi(&b->epi_b);
}
void bnn_free_react(ModelR *m) {
    for (int i = 0; i < 4; ++i) { free_block(&m->enc[i]); free_block(&m->dec[i]); }
    free_block(&m->bott);
    free(m->head_W); free(m->head_bias); free(m);
    rs_destroy();
}
