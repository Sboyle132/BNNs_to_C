/* bnn_react_blk.c - production instantiation of the register-blocked
 * real-input kernels (template: bnn_realin_blk.h) used by the react engine.
 * Config c8p8 (8 output channels x 8 pixels per register tile) won the A53
 * microbenchmark on every shape: conv 120->16 3x3 4.38x, proj 2.0-2.55x vs the
 * previous kernels (bench_real, ZCU104, single thread). Requires Cout % 8 == 0;
 * the engine falls back to the previous kernels otherwise. */
#define CB 8
#define PB 8
#define SFX _c8p8
#include "bnn_react_arena.h"
#include "bnn_realin_blk.h"

void react_proj_blk(const float *x, const float *Wm, float *out,
                    int Cin, int H, int W, int Cout) {
    proj_blk_c8p8(x, Wm, out, Cin, H, W, Cout);
}

void react_conv3_blk(const float *x, const int8_t *wsign, const float *alpha,
                     float *out, int Cin, int H, int W, int Cout) {
    conv3_blk_c8p8(x, wsign, alpha, out, Cin, H, W, Cout);
}
