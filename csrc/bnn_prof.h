/* bnn_prof.h - opt-in per-op profiler.
 *
 * Without -DBNN_PROFILE, PROF(...) expands to just the wrapped call and
 * PLABEL(...) to a null pointer, so the verified engine is byte-for-byte
 * unchanged and nothing is timed. The profiler functions are always linked
 * (prof_now is also used by main.c for coarse timing) but prof_report() is
 * silent unless records were actually added, which only happens under the flag.
 */
#ifndef BNN_PROF_H
#define BNN_PROF_H
#include <stdio.h>

double prof_now(void);                 /* CLOCK_MONOTONIC seconds */
void   prof_reset(void);
void   prof_add(const char *label, const char *optype,
                double out_elems, double taps, double sec);
void   prof_report(FILE *out);         /* per-instance rows + roll-up by op type */

#ifdef BNN_PROFILE
  #define PROF(label, optype, oe, taps, CALL) do {                    \
          double _p0 = prof_now(); CALL;                              \
          prof_add((label), (optype), (double)(oe), (double)(taps),   \
                   prof_now() - _p0);                                 \
      } while (0)
  /* format a per-instance label into buf and return it (comma operator) */
  #define PLABEL(buf, ...) (snprintf((buf), sizeof(buf), __VA_ARGS__), (buf))
#else
  /* sizeof(label) names any buffer used to build the label, so the wrapped
   * call sites need no #ifdef to avoid unused-variable warnings. */
  #define PROF(label, optype, oe, taps, CALL) do { (void)sizeof(label); CALL; } while (0)
  #define PLABEL(buf, ...) ((void)sizeof(buf), (const char *)0)
#endif

#endif
