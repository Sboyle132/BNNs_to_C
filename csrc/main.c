/* main.c - standalone BNN inference: blob + input patch -> logits.
 * usage: bnn_infer weights.blob input.bin output.bin [H W n_bands]
 * input.bin  : n_bands*H*W float32, layout [band,row,col]
 * output.bin : n_classes*H*W float32, layout [class,row,col] */
#include "bnn_model.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s weights.blob input.bin output.bin [H W]\n", argv[0]);
        return 1;
    }
    int H = (argc > 4) ? atoi(argv[4]) : 32;
    int W = (argc > 5) ? atoi(argv[5]) : 32;
    Model *m = bnn_load(argv[1]);
    long n_in = (long)m->n_bands * H * W;
    float *input = malloc(n_in * sizeof(float));
    FILE *fi = fopen(argv[2], "rb");
    if (!fi || fread(input, sizeof(float), n_in, fi) != (size_t)n_in) {
        fprintf(stderr, "bad input file\n"); return 1;
    }
    fclose(fi);
    long n_out = (long)m->n_classes * H * W;
    float *logits = malloc(n_out * sizeof(float));
    bnn_forward(m, input, H, W, logits);
    FILE *fo = fopen(argv[3], "wb");
    fwrite(logits, sizeof(float), n_out, fo);
    fclose(fo);
    free(input); free(logits); bnn_free(m);
    return 0;
}
