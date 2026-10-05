/* bnn_react_arena.h - persistent caching scratch pool for the react engine.
 * rs_* replace malloc/calloc/free on the PER-FRAME path only (not the one-time
 * load path). When enabled, allocations are served from 64-byte-aligned blocks
 * that are returned to a free-list (not the OS) on rs_free and reused next
 * frame. Same shapes every frame => zero OS allocation after frame 1 =>
 * deterministic latency, no heap churn. When disabled (default), rs_* fall back
 * to malloc/free so the benches and any other caller are unaffected. The pool
 * changes WHERE memory comes from, never any computed value. */
#ifndef BNN_REACT_ARENA_H
#define BNN_REACT_ARENA_H
#include <stddef.h>
void  rs_enable(int on);     /* engine turns the pool on; default off */
void *rs_alloc(size_t n);    /* uninitialised */
void *rs_calloc(size_t n);   /* zeroed (bytes) */
void  rs_free(void *p);
void  rs_destroy(void);      /* release all blocks to the OS */
size_t rs_footprint(void);   /* total bytes held by the pool */
int    rs_nblocks(void);     /* number of blocks (== distinct concurrent sizes) */
#endif
