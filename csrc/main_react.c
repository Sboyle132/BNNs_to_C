/* main_react.c - standalone driver for the ReActNet (v3) engine.
 *   bnn_infer_react <blob> <input.bin|rand> <output.bin> H W [reps]
 *   A file input is STREAMED strip-by-strip (full cube never resident);
 *   `rand` is generated in memory. Prefix the file with "mem:" to force the
 *   in-memory path for an A/B of peak RSS.
 * input.bin : float32 [n_bands,H,W] row-major, or the literal `rand` for a
 *             deterministic synthetic cube (timing is data-independent).
 * output.bin: float32 logits [n_classes,H,W].
 * reps      : SAME image run N times; reports median/min/max forward time
 *             (jitter smoothing, not a mean, not multiple images).
 * Same argument order as main.c / main_stem.c. */
#include "bnn_model_react.h"
#include "bnn_prof.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* xorshift + Box-Muller: deterministic approx N(0,1) cube */
static void fill_rand(float *x, size_t n) {
    uint64_t s = 88172645463325252ULL;
    for (size_t i = 0; i < n; ++i) {
        double u[2];
        for (int k = 0; k < 2; ++k) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            u[k] = ((s >> 11) + 0.5) / 9007199254740992.0;
        }
        x[i] = (float)(__builtin_sqrt(-2.0 * __builtin_log(u[0])) *
                       __builtin_cos(6.283185307179586 * u[1]));
    }
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s blob input|rand output H W [reps]\n", argv[0]);
        return 2;
    }
    const char *blob = argv[1], *inp = argv[2], *outp = argv[3];
    int H = atoi(argv[4]), W = atoi(argv[5]);
    int reps = argc > 6 ? atoi(argv[6]) : 1;
    if (H <= 0 || W <= 0 || reps <= 0) { fprintf(stderr, "bad H/W/reps\n"); return 2; }

    double t0 = now_ms();
    ModelR *m = bnn_load_react(blob);
    double t_load = now_ms() - t0;

    size_t nin = (size_t)m->n_bands * H * W, nout = (size_t)m->n_classes * H * W;
    float *out = malloc(nout * sizeof(float));
    if (!out) { fprintf(stderr, "OOM out\n"); return 1; }

    /* input spec: [patch<P>:][mem:]FILE | [patch<P>:]rand
     *   patch<P>: run patch-wise (independent PxP patches, matches 32x32 training)
     *   mem:      load the whole cube into RAM first (default for files is streaming)
     *   rand      synthetic cube, always in RAM */
    int patchP = 0;
    if (!strncmp(inp, "patch", 5)) {
        patchP = atoi(inp + 5);
        const char *q = strchr(inp, ':');
        if (patchP <= 0 || !q) { fprintf(stderr, "bad patch spec '%s' (use patch32:FILE)\n", inp); return 2; }
        if (patchP % 16) { fprintf(stderr, "patch size must be a multiple of 16\n"); return 2; }
        inp = q + 1;
    }
    int is_rand = (inp[0]=='r'&&inp[1]=='a'&&inp[2]=='n'&&inp[3]=='d'&&inp[4]==0);
    int force_mem = (inp[0]=='m'&&inp[1]=='e'&&inp[2]=='m'&&inp[3]==':');
    const char *path = force_mem ? inp + 4 : inp;

    double *ts = malloc(reps * sizeof(double));
    if (is_rand || force_mem) {
        /* in-memory path: full cube resident (313 MB at capture size) */
        float *in = malloc(nin * sizeof(float));
        if (!in) { fprintf(stderr, "OOM in\n"); return 1; }
        if (is_rand) fill_rand(in, nin);
        else { FILE *f = fopen(path, "rb");
            if (!f || fread(in, sizeof(float), nin, f) != nin) { fprintf(stderr, "bad input %s\n", path); return 1; }
            fclose(f); }
        for (int r = 0; r < reps; ++r) { double a = now_ms();
            if (patchP) bnn_forward_react_patched(m, in, H, W, out, patchP);
            else        bnn_forward_react(m, in, H, W, out);
            ts[r] = now_ms()-a; }
        free(in);
    } else {
        /* streaming path: cube read from the file in pieces, never fully resident */
        FILE *f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
        for (int r = 0; r < reps; ++r) { double a = now_ms();
            if (patchP) bnn_forward_react_patched_file(m, f, H, W, out, patchP);
            else        bnn_forward_react_file(m, f, H, W, out);
            ts[r] = now_ms()-a; }
        fclose(f);
    }
    qsort(ts, reps, sizeof(double), cmp_d);   /* sorted: median = ts[reps/2], min = ts[0], max = ts[reps-1] */

    FILE *g = fopen(outp, "wb");
    if (!g) { fprintf(stderr, "cannot open %s\n", outp); return 1; }
    fwrite(out, sizeof(float), nout, g); fclose(g);

    printf("# load %.1f ms\n", t_load);
    if (patchP) printf("# mode: patch-wise P=%d\n", patchP);
    printf("# forward %dx%d reps=%d  median %.1f ms  min %.1f  max %.1f\n",
           H, W, reps, ts[reps / 2], ts[0], ts[reps - 1]);
    prof_report(stderr);   /* per-op; silent unless built with `make -f Makefile.react profile` */
    free(ts); free(out); bnn_free_react(m);
    return 0;
}
