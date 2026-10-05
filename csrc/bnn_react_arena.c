#include "bnn_react_arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define RS_MAXBLK 2048
typedef struct { void *ptr; size_t cap; size_t used; int in_use; } Blk;
static Blk B[RS_MAXBLK];
static int N = 0, enabled = 0;
static size_t cur_used = 0, peak_used = 0;
static size_t roundup(size_t n) { return (n + 63) & ~(size_t)63; }

void rs_enable(int on) { enabled = on; }

static void *al(size_t n, int zero) {
    if (!enabled) { void *p = malloc(n ? n : 1); if (zero && p) memset(p, 0, n); return p; }
    size_t cap = roundup(n ? n : 1);
    int best = -1;
    for (int i = 0; i < N; ++i)
        if (!B[i].in_use && B[i].cap >= cap &&
            (best < 0 || B[i].cap < B[best].cap)) best = i;   /* smallest fit */
    if (best < 0) {
        if (N >= RS_MAXBLK) { fprintf(stderr, "arena: RS_MAXBLK exceeded\n"); exit(1); }
        void *p = NULL;
        if (posix_memalign(&p, 64, cap)) { fprintf(stderr, "arena OOM %zu\n", cap); exit(1); }
        B[N].ptr = p; B[N].cap = cap; B[N].in_use = 1; best = N++;
    } else B[best].in_use = 1;
    B[best].used = cap; cur_used += cap; if (cur_used > peak_used) peak_used = cur_used;
    if (zero) memset(B[best].ptr, 0, n);
    return B[best].ptr;
}
void *rs_alloc(size_t n) { return al(n, 0); }
void *rs_calloc(size_t n) { return al(n, 1); }
void rs_free(void *p) {
    if (!enabled) { free(p); return; }
    if (!p) return;
    for (int i = 0; i < N; ++i) if (B[i].ptr == p) { B[i].in_use = 0; cur_used -= B[i].used; return; }
}
void rs_destroy(void) { for (int i = 0; i < N; ++i) free(B[i].ptr); N = 0; }
size_t rs_footprint(void) { size_t s = 0; for (int i = 0; i < N; ++i) s += B[i].cap; return s; }
int rs_nblocks(void) { return N; }

/* audit: print pool blocks grouped by size (largest first) */
void rs_dump(void) {
    /* aggregate by cap */
    size_t caps[RS_MAXBLK]; int cnt[RS_MAXBLK], ng = 0;
    for (int i = 0; i < N; ++i) {
        int g = -1;
        for (int j = 0; j < ng; ++j) if (caps[j] == B[i].cap) { g = j; break; }
        if (g < 0) { caps[ng] = B[i].cap; cnt[ng] = 1; ng++; }
        else cnt[g]++;
    }
    for (int a = 0; a < ng; ++a) for (int b = a+1; b < ng; ++b)
        if (caps[b]*cnt[b] > caps[a]*cnt[a]) { size_t tc=caps[a];caps[a]=caps[b];caps[b]=tc; int ti=cnt[a];cnt[a]=cnt[b];cnt[b]=ti; }
    size_t tot = 0;
    fprintf(stderr, "# pool blocks (%d total, %.0f MB)\n", N, rs_footprint()/1e6);
    fprintf(stderr, "# %10s x %3s = %8s\n", "bytes", "cnt", "MB");
    for (int a = 0; a < ng; ++a) { fprintf(stderr, "  %10zu x %3d = %8.1f\n", caps[a], cnt[a], caps[a]*(double)cnt[a]/1e6); tot += caps[a]*(size_t)cnt[a]; }
    fprintf(stderr, "# sum %.0f MB\n", tot/1e6);
}

size_t rs_peak_used(void) { return peak_used; }
