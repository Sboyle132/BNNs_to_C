/* bnn_prof.c - profiler storage + report. Always compiled; prof_now() is used
 * by main.c for coarse timing regardless of -DBNN_PROFILE. Per-op records are
 * only added via the PROF macro, which is a no-op unless built with the flag. */
#define _POSIX_C_SOURCE 199309L
#include "bnn_prof.h"
#include <time.h>
#include <string.h>

double prof_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

#define PROF_MAX 512
typedef struct {
    char   label[48];
    char   optype[16];
    double oe;      /* output elements  */
    double taps;    /* per-output work (Cin*kh*kw for convs, else 0) */
    double sec;
} Rec;

static Rec recs[PROF_MAX];
static int nrec = 0;

void prof_reset(void) { nrec = 0; }

void prof_add(const char *label, const char *optype,
              double out_elems, double taps, double sec) {
    if (nrec >= PROF_MAX) return;
    Rec *r = &recs[nrec++];
    snprintf(r->label,  sizeof r->label,  "%s", label  ? label  : "?");
    snprintf(r->optype, sizeof r->optype, "%s", optype ? optype : "?");
    r->oe = out_elems; r->taps = taps; r->sec = sec;
}

static double rec_work(const Rec *r) {
    return r->oe * (r->taps > 0.0 ? r->taps : 1.0);
}

void prof_report(FILE *o) {
    if (nrec == 0) return;
    double tot = 0.0;
    for (int i = 0; i < nrec; ++i) tot += recs[i].sec;

    fprintf(o, "\n# per-op (execution order)\n");
    fprintf(o, "%-14s %-10s %12s %12s %10s %7s\n",
            "layer", "op", "out_elems", "work", "ms", "%");
    for (int i = 0; i < nrec; ++i) {
        const Rec *r = &recs[i];
        fprintf(o, "%-14s %-10s %12.0f %12.4g %10.4f %6.2f\n",
                r->label, r->optype, r->oe, rec_work(r),
                r->sec * 1e3, tot > 0 ? 100.0 * r->sec / tot : 0.0);
    }

    fprintf(o, "\n# roll-up by op type\n");
    fprintf(o, "%-10s %6s %12s %10s %7s\n", "op", "calls", "work", "ms", "%");
    char seen[PROF_MAX][16];
    int ns = 0;
    for (int i = 0; i < nrec; ++i) {
        int found = 0;
        for (int j = 0; j < ns; ++j)
            if (strcmp(seen[j], recs[i].optype) == 0) { found = 1; break; }
        if (!found) snprintf(seen[ns++], 16, "%s", recs[i].optype);
    }
    for (int j = 0; j < ns; ++j) {
        double s = 0, w = 0; int c = 0;
        for (int i = 0; i < nrec; ++i)
            if (strcmp(recs[i].optype, seen[j]) == 0) {
                s += recs[i].sec; w += rec_work(&recs[i]); c++;
            }
        fprintf(o, "%-10s %6d %12.4g %10.4f %6.2f\n",
                seen[j], c, w, s * 1e3, tot > 0 ? 100.0 * s / tot : 0.0);
    }
    fprintf(o, "# total per-op: %.4f ms\n", tot * 1e3);
}
