#include "bnn_react_arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define RS_MAXBLK 2048
typedef struct { void *ptr; size_t cap; int in_use; } Blk;
static Blk B[RS_MAXBLK];
static int N = 0, enabled = 0;
static size_t roundup(size_t n) { return (n + 63) & ~(size_t)63; }

void rs_enable(int on) { enabled = on; }

static void *al(size_t n, int zero) {
    if (!enabled) { void *p = malloc(n ? n : 1); if (zero && p) memset(p, 0, n); return p; }
    size_t cap = roundup(n ? n : 1);
    int best = -1;
    for (int i = 0; i < N; ++i) if (!B[i].in_use && B[i].cap == cap) { best = i; break; }
    if (best < 0) {
        if (N >= RS_MAXBLK) { fprintf(stderr, "arena: RS_MAXBLK exceeded\n"); exit(1); }
        void *p = NULL;
        if (posix_memalign(&p, 64, cap)) { fprintf(stderr, "arena OOM %zu\n", cap); exit(1); }
        B[N].ptr = p; B[N].cap = cap; B[N].in_use = 1; best = N++;
    } else B[best].in_use = 1;
    if (zero) memset(B[best].ptr, 0, n);
    return B[best].ptr;
}
void *rs_alloc(size_t n) { return al(n, 0); }
void *rs_calloc(size_t n) { return al(n, 1); }
void rs_free(void *p) {
    if (!enabled) { free(p); return; }
    if (!p) return;
    for (int i = 0; i < N; ++i) if (B[i].ptr == p) { B[i].in_use = 0; return; }
}
void rs_destroy(void) { for (int i = 0; i < N; ++i) free(B[i].ptr); N = 0; }
size_t rs_footprint(void) { size_t s = 0; for (int i = 0; i < N; ++i) s += B[i].cap; return s; }
int rs_nblocks(void) { return N; }
