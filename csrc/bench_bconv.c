/* bench_bconv.c - sweep for the react engine's binary convolutions.
 * Compares, at the model's REAL layer shapes, the two current kernels
 * (bconv_xnor dense, bconv_xnor_packed) with the candidates in
 * bnn_bconv_neon.c (bconv_packed_neon, bconv_packed_neon_b4). Answers the free
 * dispatch-threshold question and shows the per-shape winner.
 *
 *   ./bench_bconv            correctness (edge + real shapes) then timing
 *   ./bench_bconv check      correctness only
 *   ./bench_bconv time       timing only
 *
 * Correctness is EXACT (binary conv is integer): any mismatch vs
 * bconv_naive_int is a bug, not float noise. Timing is min of 3, single
 * thread. Each kernel does its own input pack + weight repack internally
 * (same as the current kernels and the profile), so GMAC/s is comparable to
 * the profile's bconv rows. Weight MACs per output = Cin*kh*kw. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* current kernels (bnn_conv.c) */
void bconv_naive_int(const int8_t *, const int8_t *, int32_t *, int, int, int, int, int, int, int, int);
void pack_weights(const int8_t *, uint64_t *, int, int, int, int);
void bconv_xnor(const int8_t *, const uint64_t *, int32_t *, int, int, int, int, int, int, int, int);
void bconv_xnor_packed(const int8_t *, const uint64_t *, int32_t *, int, int, int, int, int, int, int, int);
/* candidates (bnn_bconv_neon.c) */
void bconv_packed_neon(const int8_t *, const uint64_t *, int32_t *, int, int, int, int, int, int, int, int);
void bconv_blk_b4(const int8_t *, const uint64_t *, int32_t *, int, int, int, int, int, int, int, int);
void bconv_blk_b8(const int8_t *, const uint64_t *, int32_t *, int, int, int, int, int, int, int, int);
void bconv_blk_b16(const int8_t *, const uint64_t *, int32_t *, int, int, int, int, int, int, int, int);
void bconv_blk_b4p2(const int8_t *, const uint64_t *, int32_t *, int, int, int, int, int, int, int, int);
void bconv_blk_b8p2(const int8_t *, const uint64_t *, int32_t *, int, int, int, int, int, int, int, int);
void bconv_blk_b4p4(const int8_t *, const uint64_t *, int32_t *, int, int, int, int, int, int, int, int);

typedef void (*bk)(const int8_t *, const uint64_t *, int32_t *, int, int, int, int, int, int, int, int);
typedef struct { const char *name; bk fn; int max_cin; } Kern;   /* max_cin: skip above (dense gather explodes) */
static const Kern K[] = {
    {"packed",  bconv_xnor_packed, 100000},   /* current (Cin>=36) */
    {"b4",      bconv_blk_b4,      100000},   /* last round's winner */
    {"b8",      bconv_blk_b8,      100000},
    {"b16",     bconv_blk_b16,     100000},
    {"b4p2",    bconv_blk_b4p2,    100000},
    {"b8p2",    bconv_blk_b8p2,    100000},
    {"b4p4",    bconv_blk_b4p4,    100000},
};
#define NK (int)(sizeof K / sizeof K[0])

/* the 17 binary convs: {Cin,H,W,Cout}, k=3 pad=1 stride=1 */
typedef struct { const char *tag; int Cin, H, W, Cout; } Shape;
static const Shape SH[] = {
    {"enc1.u2", 16, 598, 1092, 16}, {"enc2.u1", 16, 299, 546, 32},
    {"enc2.u2", 32, 299, 546, 32},  {"enc3.u1", 32, 149, 273, 64},
    {"enc3.u2", 64, 149, 273, 64},  {"enc4.u1", 64, 74, 136, 128},
    {"enc4.u2", 128, 74, 136, 128}, {"bott.u1", 128, 37, 68, 256},
    {"bott.u2", 256, 37, 68, 256},  {"dec4.u1", 384, 74, 136, 128},
    {"dec4.u2", 128, 74, 136, 128}, {"dec3.u1", 192, 149, 273, 64},
    {"dec3.u2", 64, 149, 273, 64},  {"dec2.u1", 96, 299, 546, 32},
    {"dec2.u2", 32, 299, 546, 32},  {"dec1.u1", 48, 598, 1092, 16},
    {"dec1.u2", 16, 598, 1092, 16},
};
#define NSH (int)(sizeof SH / sizeof SH[0])

static uint64_t rs = 0x243F6A8885A308D3ULL;
static int8_t rsign(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 40) & 1 ? 1 : -1; }
static int8_t *rnd_pm1(size_t n) { int8_t *p = malloc(n); for (size_t i = 0; i < n; ++i) p[i] = rsign(); return p; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }

static uint64_t *packw(const int8_t *w, int Cout, int Cin, int kh, int kw, int *nwords) {
    *nwords = (Cin*kh*kw + 63) / 64;
    uint64_t *pw = malloc((size_t)Cout * *nwords * 8);
    pack_weights(w, pw, Cout, Cin, kh, kw);
    return pw;
}

static int check_shape(int Cin, int H, int W, int Cout, const char *tag) {
    size_t na = (size_t)Cin*H*W, no = (size_t)Cout*H*W;
    int8_t *a = rnd_pm1(na), *w = rnd_pm1((size_t)Cout*Cin*9);
    int nwords; uint64_t *pw = packw(w, Cout, Cin, 3, 3, &nwords);
    int32_t *ref = malloc(no*4), *got = malloc(no*4);
    bconv_naive_int(a, w, ref, Cin, H, W, Cout, 3, 3, 1, 1);
    int bad = 0;
    printf("  %-8s %3d->%-3d %3dx%-4d", tag, Cin, Cout, H, W);
    for (int k = 0; k < NK; ++k) {
        if (Cin > K[k].max_cin) { printf("  %-8s -", K[k].name); continue; }
        memset(got, 0x5A, no*4);
        K[k].fn(a, pw, got, Cin, H, W, Cout, 3, 3, 1, 1);
        int mism = memcmp(got, ref, no*4) ? 1 : 0;
        if (mism) { size_t c = 0; for (size_t i = 0; i < no; ++i) c += got[i] != ref[i]; printf("  %s:%zuMISM", K[k].name, c); bad++; }
        else printf("  %-8s ok", K[k].name);
    }
    printf("\n"); free(a); free(w); free(pw); free(ref); free(got);
    return bad;
}

static int check_all(void) {
    int bad = 0;
    printf("== correctness vs bconv_naive_int (EXACT; any mismatch = bug) ==\n");
    /* edge shapes: tiny, odd, 1-row, 1-col, Cout not mult of 4, Cin tail bits */
    const int e[][4] = {{1,1,1,4},{3,1,5,4},{5,5,1,8},{7,4,4,6},{16,3,3,16},
                        {17,5,7,8},{63,4,6,8},{64,4,6,8},{65,4,6,8},{130,3,3,8},{48,9,9,16}};
    for (size_t i = 0; i < sizeof e/sizeof e[0]; ++i) bad += check_shape(e[i][0],e[i][1],e[i][2],e[i][3],"edge");
    /* a few real shapes shrunk in H,W (full res is slow; logic is size-independent) */
    for (int s = 0; s < NSH; ++s) {
        int H = SH[s].H > 40 ? 40 : SH[s].H, W = SH[s].W > 44 ? 44 : SH[s].W;
        bad += check_shape(SH[s].Cin, H, W, SH[s].Cout, SH[s].tag);
    }
    printf("%s\n", bad ? "CORRECTNESS: FAIL" : "CORRECTNESS: PASS");
    return bad;
}

#define REPS 3
static void time_all(void) {
    printf("\n== timing at real layer shapes (min of %d reps, single thread) ==\n", REPS);
    printf("%-8s %-9s %4s %9s", "layer", "Cin->Co", "Wc", "ms");
    for (int k = 0; k < NK; ++k) printf(" %8s", K[k].name);
    printf("   best\n");
    double tot[NK]; for (int k = 0; k < NK; ++k) tot[k] = 0;
    double tot_cur = 0;                          /* current dispatch: dense if Cin<36 else packed */
    double bestmix = 0;                          /* sum of each shape's fastest kernel */
    for (int s = 0; s < NSH; ++s) {
        int Cin = SH[s].Cin, H = SH[s].H, W = SH[s].W, Cout = SH[s].Cout, Wc = (Cin+63)/64;
        size_t no = (size_t)Cout*H*W;
        int8_t *a = rnd_pm1((size_t)Cin*H*W), *w = rnd_pm1((size_t)Cout*Cin*9);
        int nwords; uint64_t *pw = packw(w, Cout, Cin, 3, 3, &nwords);
        int32_t *P = malloc(no*4);
        double ms[NK]; int bi = -1; double bms = 1e30;
        printf("%-8s %3d->%-3d %4d %9s", SH[s].tag, Cin, Cout, Wc, "");
        for (int k = 0; k < NK; ++k) {
            if (Cin > K[k].max_cin) { ms[k] = -1; printf(" %8s", "-"); continue; }
            double best = 1e30;
            for (int r = 0; r < REPS; ++r) { double t = now_ms(); K[k].fn(a, pw, P, Cin, H, W, Cout, 3, 3, 1, 1); t = now_ms()-t; if (t < best) best = t; }
            ms[k] = best; tot[k] += best; printf(" %8.1f", best);
            if (best < bms) { bms = best; bi = k; }
        }
        tot_cur += ms[0];   /* current engine uses packed for all react bconv shapes */
        bestmix += bms;
        printf("   %s\n", bi >= 0 ? K[bi].name : "?");
        free(a); free(w); free(pw); free(P);
    }
    printf("%-8s %-9s %4s %9s", "TOTAL", "", "", "");
    for (int k = 0; k < NK; ++k) { if (tot[k] > 0) printf(" %8.1f", tot[k]); else printf(" %8s", "-"); }
    printf("\n");
    printf("current dispatch   (dense if Cin<36 else packed) : %8.1f ms\n", tot_cur);
    printf("best-per-shape mix (lower bound if we dispatch to the winner): %8.1f ms\n", bestmix);
}

int main(int argc, char **argv) {
    int ck = 1, tm = 1;
    if (argc > 1 && !strcmp(argv[1], "check")) tm = 0;
    if (argc > 1 && !strcmp(argv[1], "time")) ck = 0;
    if (ck && check_all()) return 1;
    if (tm) time_all();
    return 0;
}
