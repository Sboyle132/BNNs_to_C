/* main.c - standalone BNN inference: blob + input patch -> logits.
 * usage: bnn_infer weights.blob input.bin|rand output.bin [H W [reps]]
 *   input.bin : n_bands*H*W float32, layout [band,row,col]; "rand" = synthetic
 *   output.bin: n_classes*H*W float32, layout [class,row,col]
 *   reps      : run forward N times, report median/min/max (default 1)
 *
 * Coarse timing (blob load / forward / write) always prints to stderr.
 * Per-op breakdown also prints when built with `make profile` (-DBNN_PROFILE);
 * it is silent otherwise. Timing is data-independent for the fixed-loop
 * kernels, so "rand" input at real padded dims benchmarks without a capture. */
#include "bnn_model.h"
#include "bnn_prof.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s weights.blob input.bin|rand output.bin [H W [reps]]\n", argv[0]);
        return 1;
    }
    int H = (argc > 4) ? atoi(argv[4]) : 32;
    int W = (argc > 5) ? atoi(argv[5]) : 32;
    int reps = (argc > 6) ? atoi(argv[6]) : 1;
    if (reps < 1) reps = 1;

    double t0 = prof_now();
    Model *m = bnn_load(argv[1]);
    double t_load = prof_now() - t0;

    long n_in = (long)m->n_bands * H * W;
    float *input = malloc(n_in * sizeof(float));
    if (!input) { fprintf(stderr, "OOM input\n"); return 1; }
    if (strcmp(argv[2], "rand") == 0) {
        srand(1234);
        for (long i = 0; i < n_in; ++i) input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    } else {
        FILE *fi = fopen(argv[2], "rb");
        if (!fi || fread(input, sizeof(float), n_in, fi) != (size_t)n_in) {
            fprintf(stderr, "bad input file\n"); return 1;
        }
        fclose(fi);
    }

    long n_out = (long)m->n_classes * H * W;
    float *logits = malloc(n_out * sizeof(float));
    double *ts = malloc((size_t)reps * sizeof(double));
    if (!logits || !ts) { fprintf(stderr, "OOM\n"); return 1; }

    for (int r = 0; r < reps; ++r) {
        double f0 = prof_now();
        bnn_forward(m, input, H, W, logits);
        ts[r] = (prof_now() - f0) * 1e3;   /* ms */
    }
    qsort(ts, reps, sizeof(double), cmp_d);
    double med = ts[reps / 2], tmin = ts[0], tmax = ts[reps - 1];

    double w0 = prof_now();
    FILE *fo = fopen(argv[3], "wb");
    if (!fo) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
    fwrite(logits, sizeof(float), n_out, fo);
    fclose(fo);
    double t_write = (prof_now() - w0) * 1e3;

    fprintf(stderr, "# H=%d W=%d bands=%d classes=%d reps=%d\n",
            H, W, m->n_bands, m->n_classes, reps);
    fprintf(stderr, "# blob load : %9.3f ms\n", t_load * 1e3);
    fprintf(stderr, "# forward   : %9.3f ms  (median; min %.3f, max %.3f)\n", med, tmin, tmax);
    fprintf(stderr, "# out write : %9.3f ms\n", t_write);

    prof_report(stderr);   /* per-op; silent unless built with `make profile` */

    free(ts); free(input); free(logits); bnn_free(m);
    return 0;
}
