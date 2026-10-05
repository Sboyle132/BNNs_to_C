/* mem_probe.c - engine memory report. Runs N full-size frames and prints the
 * pool's held footprint, the true peak of concurrently live scratch, block
 * count, and whether they stay constant across frames (steady state).
 *   ./mem_probe weights_react.blob [H W frames]     default 598 1092 3 */
#include <stdio.h>
#include <stdlib.h>
#include "bnn_model_react.h"
#include "bnn_react_arena.h"
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s blob [H W frames]\n", argv[0]); return 2; }
    int H = argc > 2 ? atoi(argv[2]) : 598, W = argc > 3 ? atoi(argv[3]) : 1092;
    int F = argc > 4 ? atoi(argv[4]) : 3;
    ModelR *m = bnn_load_react(argv[1]);
    float *in = calloc((size_t)m->n_bands * H * W, 4), *o = calloc((size_t)m->n_classes * H * W, 4);
    if (!in || !o) { fprintf(stderr, "OOM io\n"); return 1; }
    for (int f = 0; f < F; ++f) {
        bnn_forward_react(m, in, H, W, o);
        printf("frame %d: pool held %6.1f MB   peak live %6.1f MB   blocks %d\n",
               f, rs_footprint() / 1e6, rs_peak_used() / 1e6, rs_nblocks());
    }
    printf("input %.0f MB + output %.1f MB are caller-owned (not in pool)\n",
           (double)m->n_bands * H * W * 4 / 1e6, (double)m->n_classes * H * W * 4 / 1e6);
    bnn_free_react(m);
    return 0;
}
