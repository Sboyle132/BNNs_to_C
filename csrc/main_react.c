/* main_react.c - standalone driver for the ReActNet (v3) engine.
 *   bnn_infer_react <blob> <input.bin|rand> <output.bin> H W [reps]
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
    float *in = malloc(nin * sizeof(float)), *out = malloc(nout * sizeof(float));
    if (!in || !out) { fprintf(stderr, "OOM io\n"); return 1; }
    if (inp[0] == 'r' && inp[1] == 'a' && inp[2] == 'n' && inp[3] == 'd' && inp[4] == 0) {
        fill_rand(in, nin);
    } else {
        FILE *f = fopen(inp, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", inp); return 1; }
        if (fread(in, sizeof(float), nin, f) != nin) {
            fprintf(stderr, "input %s: expected %zu floats\n", inp, nin); return 1;
        }
        fclose(f);
    }

    double *ts = malloc(reps * sizeof(double));
    for (int r = 0; r < reps; ++r) {
        double a = now_ms();
        bnn_forward_react(m, in, H, W, out);
        ts[r] = now_ms() - a;
    }
    qsort(ts, reps, sizeof(double), cmp_d);

    FILE *g = fopen(outp, "wb");
    if (!g) { fprintf(stderr, "cannot open %s\n", outp); return 1; }
    fwrite(out, sizeof(float), nout, g); fclose(g);

    printf("# load %.1f ms\n", t_load);
    printf("# forward %dx%d reps=%d  median %.1f ms  min %.1f  max %.1f\n",
           H, W, reps, ts[reps / 2], ts[0], ts[reps - 1]);
    prof_report(stderr);   /* per-op; silent unless built with `make -f Makefile.react profile` */
    free(ts); free(in); free(out); bnn_free_react(m);
    return 0;
}