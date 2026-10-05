/* bench_epi.c - track 2: scalar epilogue (current) vs NEON, at the 22 real epi
 * shapes. Elementwise/memory-bound; one int row (weighted to the 21 int calls)
 * + a real-variant check. Correctness: rel diff vs current (fma noise ~1e-6). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void react_epilogue_int(const int32_t *, const float *, float *, int, int, int, const float *, const float *, const float *, const float *);
void react_epilogue_real(const float *, const float *, float *, int, int, int, const float *, const float *, const float *, const float *);
void react_epilogue_int_v2(const int32_t *, const float *, float *, int, int, int, const float *, const float *, const float *, const float *);
void react_epilogue_real_v2(const float *, const float *, float *, int, int, int, const float *, const float *, const float *, const float *);

typedef struct { int C, H, W, n; } Shape;   /* n = how many of the 22 calls share this shape */
static const Shape SH[] = {
    {16, 598, 1092, 3}, {16, 299, 546, 1}, {32, 299, 546, 4}, {32, 149, 273, 1},
    {64, 149, 273, 4},  {64, 74, 136, 1},  {128, 74, 136, 4},  {128, 37, 68, 1},
    {256, 37, 68, 2},   /* + enc1.u1 is the 1 REAL call @16x598x1092, timed separately */
};
#define NSH (int)(sizeof SH / sizeof SH[0])

static uint64_t rs = 0x9E3779B97F4A7C15ULL;
static float rf1(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return (float)((rs>>40)/8388608.0-1.0);}
static float *rf(size_t n){float*p=malloc(n*4);for(size_t i=0;i<n;++i)p[i]=rf1();return p;}
static int32_t *ri(size_t n){int32_t*p=malloc(n*4);for(size_t i=0;i<n;++i)p[i]=(int32_t)(rf1()*30);return p;}
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static double reld(const float*a,const float*b,size_t n){double d=0,m=1e-30;for(size_t i=0;i<n;++i){double e=fabs((double)a[i]-b[i]);if(e>d)d=e;double r=fabs((double)b[i]);if(r>m)m=r;}return d/m;}
#define REPS 3

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    int bad = 0; double scur = 0, snew = 0;
    printf("== epi: NEON vs scalar (current), real shapes ==\n");
    printf("%-14s %10s %10s %7s %8s\n", "shape(xN)", "scalar_ms", "neon_ms", "speedup", "reldiff");
    for (int s = 0; s < NSH; ++s) {
        int C = SH[s].C, H = SH[s].H, W = SH[s].W; size_t n = (size_t)C*H*W;
        int32_t *P = ri(n); float *res = rf(n), *A = rf(C), *B = rf(C), *m1 = rf(C), *pw = rf(C);
        float *o1 = malloc(n*4), *o2 = malloc(n*4);
        react_epilogue_int(P, res, o1, C, H, W, A, B, m1, pw);
        react_epilogue_int_v2(P, res, o2, C, H, W, A, B, m1, pw);
        double d = reld(o2, o1, n); if (d > 1e-4) bad++;
        double tc=1e30,tn=1e30;
        for(int r=0;r<REPS;++r){double t=ms();react_epilogue_int(P,res,o1,C,H,W,A,B,m1,pw);t=ms()-t;if(t<tc)tc=t;}
        for(int r=0;r<REPS;++r){double t=ms();react_epilogue_int_v2(P,res,o2,C,H,W,A,B,m1,pw);t=ms()-t;if(t<tn)tn=t;}
        char tag[20]; snprintf(tag,sizeof tag,"%dx%dx%d x%d",C,H,W,SH[s].n);
        printf("%-14s %10.2f %10.2f %6.2fx %8.0e\n", tag, tc*SH[s].n, tn*SH[s].n, tc/tn, d);
        scur += tc*SH[s].n; snew += tn*SH[s].n;
        free(P);free(res);free(A);free(B);free(m1);free(pw);free(o1);free(o2);
    }
    /* the 1 real epilogue (enc1.u1) */
    { int C=16,H=598,W=1092; size_t n=(size_t)C*H*W;
      float *cin=rf(n),*res=rf(n),*A=rf(C),*B=rf(C),*m1=rf(C),*pw=rf(C),*o1=malloc(n*4),*o2=malloc(n*4);
      react_epilogue_real(cin,res,o1,C,H,W,A,B,m1,pw); react_epilogue_real_v2(cin,res,o2,C,H,W,A,B,m1,pw);
      double d=reld(o2,o1,n); if(d>1e-4)bad++;
      double tc=1e30,tn=1e30;
      for(int r=0;r<REPS;++r){double t=ms();react_epilogue_real(cin,res,o1,C,H,W,A,B,m1,pw);t=ms()-t;if(t<tc)tc=t;}
      for(int r=0;r<REPS;++r){double t=ms();react_epilogue_real_v2(cin,res,o2,C,H,W,A,B,m1,pw);t=ms()-t;if(t<tn)tn=t;}
      printf("%-14s %10.2f %10.2f %6.2fx %8.0e  (real)\n","16x598x1092 x1",tc,tn,tc/tn,d);
      scur+=tc; snew+=tn; free(cin);free(res);free(A);free(B);free(m1);free(pw);free(o1);free(o2); }
    printf("TOTAL epi: scalar %.1f ms -> neon %.1f ms  (%.2fx)\n", scur, snew, scur/snew);
    printf("%s\n", bad?"CORRECTNESS: FAIL":"CORRECTNESS: PASS");
    return bad?1:0;
}
