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
 *           down = epi_down(maxpool_P_i16(P2), maxpool_real_v2(h1))
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
static void bconv_react(const ConvR *c, const int8_t *a, int16_t *P, int H, int W) {
    /* every ReActNet binary conv is 3x3/pad1/stride1; no int16 fallback kernel needed */
    if (c->kh == 3 && c->kw == 3 && c->pad == 1 && c->stride == 1)
        bconv_blk_b4p4_i16(a, c->packed, P, c->Cin, H, W, c->Cout, 3, 3, 1, 1);
    else { fprintf(stderr, "bconv_react: unexpected %dx%d conv\n", c->kh, c->kw); exit(1); }
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
static int16_t *bconv_apply(const ConvR *c, const int8_t *a, int H, int W, const char *tag) {
    char lb[48]; (void)tag; (void)lb;
    int16_t *P = rs_alloc((size_t)c->Cout * H * W * sizeof(int16_t));
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
    int16_t *P1 = bconv_apply(&b->conv1, a1, H, W, tag);
    rs_free(a1);
    PROF(PLABEL(lb, "%s.epi", tag), "epi",
         (double)b->conv1.Cout * H * W, 0.0,
         react_epilogue_int_i16(P1, res, res, b->conv1.Cout, H, W,
                            b->epi1.A, b->epi1.B, b->epi1.move1, b->epi1.prelu_w));
    rs_free(P1);
    return res;
}

/* unit2 for a plain block: consumes h1, returns h2 (written in place) */
static float *unit2_plain(const BlockR *b, float *h1, int H, int W, const char *tag) {
    char lb[48]; (void)lb;
    int8_t *a2 = tap_apply(&b->tap2, h1, H, W, tag);
    int16_t *P2 = bconv_apply(&b->conv2, a2, H, W, tag);
    rs_free(a2);
    PROF(PLABEL(lb, "%s.epi", tag), "epi",
         (double)b->conv2.Cout * H * W, 0.0,
         react_epilogue_int_i16(P2, h1, h1, b->conv2.Cout, H, W,
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
    int16_t *P2 = bconv_apply(&b->conv2, a2, H, W, t2);
    rs_free(a2);

    float *skip = rs_alloc((size_t)Co * H * W * sizeof(float));
    PROF(PLABEL(lb, "%s.skip", tag), "epi",
         (double)Co * H * W, 0.0,
         react_epilogue_int_i16(P2, h1, skip, Co, H, W,
                            b->epi_a.A, b->epi_a.B, b->epi_a.move1, b->epi_a.prelu_w));

    int Hh = H / 2, Wh = W / 2;
    int16_t *Pd = rs_alloc((size_t)Co * Hh * Wh * sizeof(int16_t));
    PROF(PLABEL(lb, "%s.pool", tag), "maxpool",
         (double)Co * Hh * Wh, 4.0,
         maxpool_P_i16(P2, Pd, Co, H, W, 2, 2));
    rs_free(P2);
    float *h1d = rs_alloc((size_t)Co * Hh * Wh * sizeof(float));
    PROF(PLABEL(lb, "%s.poolr", tag), "maxpool",
         (double)Co * Hh * Wh, 4.0,
         maxpool_real_v2(h1, h1d, Co, H, W, 2, 2));
    rs_free(h1);
    PROF(PLABEL(lb, "%s.down", tag), "epi",
         (double)Co * Hh * Wh, 0.0,
         react_epilogue_int_i16(Pd, h1d, h1d, Co, Hh, Wh,
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

/* ---- streaming input for enc1 (don't hold the full cube) ---------------------
 * enc1 is the only consumer of the raw input and is local (two 3x3 convs + a
 * 2x2 pool, RF radius 2). So it is driven in row-strips with a small halo; each
 * strip's valid output rows are bit-exact (overlap-save). The input rows for a
 * strip come from a reader callback, so the caller decides whether the cube is
 * streamed from a file (never fully resident) or indexed from memory. */
typedef void (*bnn_in_reader)(void *ctx, int r0, int r1, int W, int nb, float *dst);

typedef struct { const float *data; int H; } mem_ctx;
static void mem_reader(void *vc, int r0, int r1, int W, int nb, float *dst) {
    const mem_ctx *c = vc;
    for (int b = 0; b < nb; ++b)
        memcpy(dst + (size_t)b * (r1 - r0) * W,
               c->data + ((size_t)b * c->H + r0) * W, (size_t)(r1 - r0) * W * sizeof(float));
}

#define ENC1_STRIP 64          /* default output rows per strip (even); board: speed flat vs S, S=64 packs the pool best (lowest RSS) */
/* runtime override for tuning: BNN_ENC1_STRIP=<even rows>. Any even value >= 8
 * gives bit-identical output; it only trades halo recompute vs buffer size. */
static int enc1_strip_rows(void) {
    const char *e = getenv("BNN_ENC1_STRIP");
    int v = e ? atoi(e) : ENC1_STRIP;
    if (v < 8) v = ENC1_STRIP;
    return v & ~1;
}
#define ENC1_HALO  2           /* enc1 = two stacked 3x3 convs -> RF radius 2; even for pool alignment */

/* produce full skip[0] (H x W) and down (H/2 x W/2) by strip-processing enc1 */
static void enc1_strided(const ModelR *m, bnn_in_reader rd, void *ctx, int H, int W,
                         float **skip0, float **down0, int *downC) {
    const BlockR *b = &m->enc[0];
    int Co = b->conv2.Cout, Hh = H / 2, Wh = W / 2, nb = m->n_bands;
    float *skip = rs_alloc((size_t)Co * H * W * sizeof(float));
    float *down = rs_alloc((size_t)Co * Hh * Wh * sizeof(float));
    const int S = enc1_strip_rows();
    for (int y0 = 0; y0 < H; y0 += S) {
        int y1 = y0 + S < H ? y0 + S : H;                            /* even */
        int r0 = y0 - ENC1_HALO > 0 ? y0 - ENC1_HALO : 0;            /* even */
        int r1 = y1 + ENC1_HALO < H ? y1 + ENC1_HALO : H;            /* even */
        int sh = r1 - r0;
        float *in_s = rs_alloc((size_t)nb * sh * W * sizeof(float));
        rd(ctx, r0, r1, W, nb, in_s);
        float *sk_s, *dn_s;
        down_block(b, in_s, sh, W, &sk_s, &dn_s, "enc1");            /* reuses the block kernel */
        rs_free(in_s);
        for (int c = 0; c < Co; ++c)                                 /* valid skip rows [y0,y1) */
            memcpy(skip + ((size_t)c * H + y0) * W,
                   sk_s + ((size_t)c * sh + (y0 - r0)) * W, (size_t)(y1 - y0) * W * sizeof(float));
        int dy0 = y0 / 2, dy1 = y1 / 2, dsrc = (y0 - r0) / 2, dsh = sh / 2;
        for (int c = 0; c < Co; ++c)                                 /* valid down rows [y0/2,y1/2) */
            memcpy(down + ((size_t)c * Hh + dy0) * Wh,
                   dn_s + ((size_t)c * dsh + dsrc) * Wh, (size_t)(dy1 - dy0) * Wh * sizeof(float));
        rs_free(sk_s); rs_free(dn_s);
    }
    *skip0 = skip; *down0 = down; *downC = Co;
}

/* dec1 + head, strip-processed (mirror of enc1): dec1 is full-res and local
 * (two 3x3 convs, RF radius 2), so its 48-channel concat is built per-strip
 * instead of whole. skip0 (16ch @ HxW) and xd (dec2 output, held) are indexed;
 * everything dec1-local is per-strip. Valid rows are bit-exact (overlap-save,
 * halo 2). head (1x1) is fused in so logits are written directly. */
static void dec1_strided(const ModelR *m, float *skip0, int sC,
                         float *xd, int xC, int xH, int xW,
                         int H, int W, float *logits) {
    const BlockR *b = &m->dec[3];
    int Wh = W / 2, S = enc1_strip_rows(); (void)xW;
    for (int y0 = 0; y0 < H; y0 += S) {
        int y1 = y0 + S < H ? y0 + S : H;                 /* even */
        int r0 = y0 - ENC1_HALO > 0 ? y0 - ENC1_HALO : 0; /* even */
        int r1 = y1 + ENC1_HALO < H ? y1 + ENC1_HALO : H; /* even */
        int sh = r1 - r0;
        float *sk_s = rs_alloc((size_t)sC * sh * W * sizeof(float));
        for (int c = 0; c < sC; ++c)
            memcpy(sk_s + (size_t)c * sh * W, skip0 + ((size_t)c * H + r0) * W, (size_t)sh * W * sizeof(float));
        int xr0 = r0 / 2, xrh = sh / 2;                   /* nearest-up: cat row r <- xd row r/2 */
        float *xd_s = rs_alloc((size_t)xC * xrh * Wh * sizeof(float));
        for (int c = 0; c < xC; ++c)
            memcpy(xd_s + (size_t)c * xrh * Wh, xd + ((size_t)c * xH + xr0) * Wh, (size_t)xrh * Wh * sizeof(float));
        int catC;
        float *cat = up_cat_f(sk_s, sC, sh, W, xd_s, xC, xrh, Wh, &catC);
        rs_free(sk_s); rs_free(xd_s);
        float *h1 = unit1_bin(b, cat, sh, W, "dec1"); rs_free(cat);
        float *h2 = unit2_plain(b, h1, sh, W, "dec1");
        float *lg = rs_alloc((size_t)m->head_Cout * sh * W * sizeof(float));
        head_1x1_v2(h2, m->head_W, m->head_bias, lg, m->head_Cin, sh, W, m->head_Cout);
        rs_free(h2);
        int vr0 = y0 - r0;
        for (int c = 0; c < m->head_Cout; ++c)
            memcpy(logits + ((size_t)c * H + y0) * W, lg + ((size_t)c * sh + vr0) * W, (size_t)(y1 - y0) * W * sizeof(float));
        rs_free(lg);
    }
}

/* core forward, enc1 fed by a reader. */
static void forward_core(const ModelR *m, bnn_in_reader rd, void *ctx, int H, int W, float *logits) {
    prof_reset();
    float *skip[4]; int skipC[4], skipH[4], skipW[4];
    char lbl[24];

    /* ---- encoder: enc1 strip-streamed, enc2..4 as before ---- */
    float *x; int xC, xH, xW;
    enc1_strided(m, rd, ctx, H, W, &skip[0], &x, &skipC[0]);
    skipH[0] = H; skipW[0] = W;
    xC = skipC[0]; xH = H / 2; xW = W / 2;
    for (int i = 1; i < 4; ++i) {
        float *nx;
        down_block(&m->enc[i], x, xH, xW, &skip[i], &nx, PLABEL(lbl, "enc%d", i + 1));
        rs_free(x); x = nx;
        skipC[i] = m->enc[i].conv2.Cout; skipH[i] = xH; skipW[i] = xW;
        xC = skipC[i]; xH /= 2; xW /= 2;
    }
    /* ---- bottleneck (fc) ---- */
    {
        float *h1 = unit1_bin(&m->bott, x, xH, xW, "bott.u1"); rs_free(x);
        x = unit2_plain(&m->bott, h1, xH, xW, "bott.u2");
        xC = m->bott.conv2.Cout;
    }
    /* ---- decoder dec4..dec2 (full); dec1 + head strip-streamed ---- */
    for (int i = 0; i < 3; ++i) {
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
    /* dec1 (si=0, full res) + head, strip-streamed */
    dec1_strided(m, skip[0], skipC[0], x, xC, xH, xW, H, W, logits);
    rs_free(x); rs_free(skip[0]);
}

/* in-memory API (unchanged signature): enc1 still strip-processed, but the cube
 * is indexed from the caller's buffer, so this does not reduce input residency.
 * Bit-identical to the pre-streaming engine. */
void bnn_forward_react(const ModelR *m, const float *input, int H, int W, float *logits) {
    mem_ctx c = { input, H };
    forward_core(m, mem_reader, &c, H, W, logits);
}

/* patch-wise forward: run the full network independently on each PxP patch
 * (zero-padded at image borders by the convs), exactly as a standalone PxP
 * image, and stitch. This reproduces 32x32 patch inference (how the model was
 * trained), NOT whole-capture inference; the two differ at patch seams because
 * the receptive field (92) exceeds the patch. Memory = one patch working set,
 * reused across patches. P must be a multiple of 16 (four 2x2 pools). */
void bnn_forward_react_patched(const ModelR *m, const float *input, int H, int W,
                               float *logits, int P) {
    int nb = m->n_bands, ncl = m->n_classes;
    float *pin = rs_alloc((size_t)nb * P * P * sizeof(float));
    float *pout = rs_alloc((size_t)ncl * P * P * sizeof(float));
    for (int py = 0; py < H; py += P) {
        int ph = py + P < H ? P : H - py;
        for (int px = 0; px < W; px += P) {
            int pw = px + P < W ? P : W - px;
            if (ph < P || pw < P) memset(pin, 0, (size_t)nb * P * P * sizeof(float));
            for (int b = 0; b < nb; ++b)
                for (int y = 0; y < ph; ++y)
                    memcpy(pin + ((size_t)b * P + y) * P,
                           input + ((size_t)b * H + py + y) * W + px, (size_t)pw * sizeof(float));
            bnn_forward_react(m, pin, P, P, pout);        /* full net on one PxP patch */
            for (int c = 0; c < ncl; ++c)
                for (int y = 0; y < ph; ++y)
                    memcpy(logits + ((size_t)c * H + py + y) * W + px,
                           pout + ((size_t)c * P + y) * P, (size_t)pw * sizeof(float));
        }
    }
    rs_free(pin); rs_free(pout);
}

/* patch-wise forward reading the cube from a file ([n_bands,H,W] f32). One
 * patch-ROW at a time: for each band the P rows of the current patch row are
 * read contiguously (P*W floats), then sliced into PxP patches. Resident input
 * = nb*P*W floats (17 MB at 120x32x1092), not the whole cube. */
void bnn_forward_react_patched_file(const ModelR *m, FILE *f, int H, int W,
                                    float *logits, int P) {
    int nb = m->n_bands, ncl = m->n_classes;
    float *rowbuf = rs_alloc((size_t)nb * P * W * sizeof(float));
    float *pin = rs_alloc((size_t)nb * P * P * sizeof(float));
    float *pout = rs_alloc((size_t)ncl * P * P * sizeof(float));
    for (int py = 0; py < H; py += P) {
        int ph = py + P < H ? P : H - py;
        for (int b = 0; b < nb; ++b) {
            long off = ((long)b * H + py) * W * (long)sizeof(float);
            if (fseek(f, off, SEEK_SET) != 0 ||
                fread(rowbuf + (size_t)b * P * W, sizeof(float), (size_t)ph * W, f) != (size_t)ph * W) {
                fprintf(stderr, "patched_file: short read band %d rows %d..%d\n", b, py, py + ph); exit(1);
            }
        }
        for (int px = 0; px < W; px += P) {
            int pw = px + P < W ? P : W - px;
            if (ph < P || pw < P) memset(pin, 0, (size_t)nb * P * P * sizeof(float));
            for (int b = 0; b < nb; ++b)
                for (int y = 0; y < ph; ++y)
                    memcpy(pin + ((size_t)b * P + y) * P,
                           rowbuf + ((size_t)b * P + y) * W + px, (size_t)pw * sizeof(float));
            bnn_forward_react(m, pin, P, P, pout);
            for (int c = 0; c < ncl; ++c)
                for (int y = 0; y < ph; ++y)
                    memcpy(logits + ((size_t)c * H + py + y) * W + px,
                           pout + ((size_t)c * P + y) * P, (size_t)pw * sizeof(float));
        }
    }
    rs_free(rowbuf); rs_free(pin); rs_free(pout);
}

/* streaming API: the cube is read strip-by-strip from a file (row-major
 * [n_bands, H, W] float32), so the full 313 MB input is never resident. */
typedef struct { FILE *f; int H; } file_ctx;
static void file_reader(void *vc, int r0, int r1, int W, int nb, float *dst) {
    file_ctx *c = vc;
    for (int b = 0; b < nb; ++b) {
        long off = ((long)b * c->H + r0) * W * (long)sizeof(float);
        if (fseek(c->f, off, SEEK_SET) != 0 ||
            fread(dst + (size_t)b * (r1 - r0) * W, sizeof(float), (size_t)(r1 - r0) * W, c->f)
                != (size_t)(r1 - r0) * W) {
            fprintf(stderr, "file_reader: short read band %d rows %d..%d\n", b, r0, r1); exit(1);
        }
    }
}
void bnn_forward_react_file(const ModelR *m, FILE *in_f, int H, int W, float *logits) {
    file_ctx c = { in_f, H };
    forward_core(m, file_reader, &c, H, W, logits);
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
