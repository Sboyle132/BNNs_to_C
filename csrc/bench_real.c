/* bench_real.c - re-sweep of the real-input kernels (proj 1x1, conv_real 3x3)
 * across CBxPB register-tile configs, at the model's REAL layer shapes.
 * conv_real is one layer (enc1, 120->16 3x3 @ full res); proj is the 9 block
 * shortcut 1x1s. Baselines: conv_realin_naive / react_proj_1x1 (current engine
 * kernels). Correctness is float-tol vs those (summation-order noise ~1e-6).
 *   ./bench_real [check|time]
 * Hypothesis: real convs favour channel tiling (reuse input vector across
 * output channels), opposite to bconv's pixel tiling. Measured, not assumed. */
#include "bnn_react_arena.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void conv_realin_naive(const float *, const int8_t *, const float *, float *, int, int, int, int, int, int, int, int);
void react_proj_1x1(const float *, const float *, float *, int, int, int, int);

#define CB 4
#define PB 8
#define SFX _c4p8
#include "bnn_realin_blk.h"
#define CB 8
#define PB 4
#define SFX _c8p4
#include "bnn_realin_blk.h"
#define CB 4
#define PB 16
#define SFX _c4p16
#include "bnn_realin_blk.h"
#define CB 8
#define PB 8
#define SFX _c8p8
#include "bnn_realin_blk.h"
#define CB 16
#define PB 4
#define SFX _c16p4
#include "bnn_realin_blk.h"
#define CB 8
#define PB 16
#define SFX _c8p16
#include "bnn_realin_blk.h"

typedef void (*pf)(const float *, const float *, float *, int, int, int, int);
typedef void (*cf)(const float *, const int8_t *, const float *, float *, int, int, int, int);
typedef struct { const char *name; int cb; pf proj; cf conv; } Cfg;
static const Cfg C[] = {
    {"c4p8",  4,  proj_blk_c4p8,  conv3_blk_c4p8},
    {"c8p4",  8,  proj_blk_c8p4,  conv3_blk_c8p4},
    {"c4p16", 4,  proj_blk_c4p16, conv3_blk_c4p16},
    {"c8p8",  8,  proj_blk_c8p8,  conv3_blk_c8p8},
    {"c16p4", 16, proj_blk_c16p4, conv3_blk_c16p4},
    {"c8p16", 8,  proj_blk_c8p16, conv3_blk_c8p16},
};
#define NC (int)(sizeof C / sizeof C[0])

typedef struct { const char *tag; int Cin, H, W, Cout; } Shape;
static const Shape PROJ[] = {
    {"enc1", 120, 598, 1092, 16}, {"enc2", 16, 299, 546, 32},
    {"enc3", 32, 149, 273, 64},   {"enc4", 64, 74, 136, 128},
    {"bott", 128, 37, 68, 256},   {"dec4", 384, 74, 136, 128},
    {"dec3", 192, 149, 273, 64},  {"dec2", 96, 299, 546, 32},
    {"dec1", 48, 598, 1092, 16},
};
#define NP (int)(sizeof PROJ / sizeof PROJ[0])
static const Shape CONV = {"enc1", 120, 598, 1092, 16};

static uint64_t rs = 0x2545F4914F6CDD1DULL;
static float rndf(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 40) / 8388608.0 - 1.0); }
static float *rf(size_t n) { float *p = malloc(n*4); for (size_t i=0;i<n;++i) p[i]=rndf(); return p; }
static int8_t *rw(size_t n) { int8_t *p = malloc(n); for (size_t i=0;i<n;++i) p[i]=rndf()>=0?1:-1; return p; }
static double ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static double reld(const float *a, const float *b, size_t n){double d=0,m=1e-30;for(size_t i=0;i<n;++i){double e=fabs((double)a[i]-b[i]);if(e>d)d=e;double r=fabs((double)b[i]);if(r>m)m=r;}return d/m;}

static int check(void) {
    int bad=0; printf("== correctness vs current kernels (rel diff, want < 1e-4) ==\n");
    const int cs[][4]={{120,16,16,16},{7,5,17,8},{3,1,40,8},{24,13,50,8},{16,4,1,16},{9,7,34,16}};
    for(size_t s=0;s<sizeof cs/sizeof cs[0];++s){int Cin=cs[s][0],H=cs[s][1],W=cs[s][2],Co=cs[s][3];
        float*x=rf((size_t)Cin*H*W),*al=rf(Co);int8_t*w=rw((size_t)Co*Cin*9);
        float*ref=malloc((size_t)Co*H*W*4),*got=malloc((size_t)Co*H*W*4);
        conv_realin_naive(x,w,al,ref,Cin,H,W,Co,3,3,1,1);
        printf("  conv %3d->%-3d %2dx%-3d",Cin,Co,H,W);
        for(int k=0;k<NC;++k){if(Co%C[k].cb){printf(" %s -",C[k].name);continue;}memset(got,0xAB,(size_t)Co*H*W*4);C[k].conv(x,w,al,got,Cin,H,W,Co);double d=reld(got,ref,(size_t)Co*H*W);printf(" %s %.0e",C[k].name,d);if(d>1e-4)bad++;}
        printf("\n");free(x);free(al);free(w);free(ref);free(got);}
    const int ps[][4]={{120,64,64,16},{48,33,47,16},{384,8,8,128},{5,3,3,8}};
    for(size_t s=0;s<sizeof ps/sizeof ps[0];++s){int Cin=ps[s][0],H=ps[s][1],W=ps[s][2],Co=ps[s][3];
        float*x=rf((size_t)Cin*H*W),*Wm=rf((size_t)Co*Cin);
        float*ref=malloc((size_t)Co*H*W*4),*got=malloc((size_t)Co*H*W*4);
        react_proj_1x1(x,Wm,ref,Cin,H,W,Co);
        printf("  proj %3d->%-3d %2dx%-3d",Cin,Co,H,W);
        for(int k=0;k<NC;++k){if(Co%C[k].cb){printf(" %s -",C[k].name);continue;}memset(got,0xAB,(size_t)Co*H*W*4);C[k].proj(x,Wm,got,Cin,H,W,Co);double d=reld(got,ref,(size_t)Co*H*W);printf(" %s %.0e",C[k].name,d);if(d>1e-4)bad++;}
        printf("\n");free(x);free(Wm);free(ref);free(got);}
    printf("%s\n",bad?"CORRECTNESS: FAIL":"CORRECTNESS: PASS"); return bad;
}

#define REPS 3
static double t_proj(pf f,const float*x,const float*Wm,float*o,int Cin,int H,int W,int Co){double b=1e30;for(int r=0;r<REPS;++r){double t=ms();f(x,Wm,o,Cin,H,W,Co);t=ms()-t;if(t<b)b=t;}return b;}
static double t_conv(cf f,const float*x,const int8_t*w,const float*al,float*o,int Cin,int H,int W,int Co){double b=1e30;for(int r=0;r<REPS;++r){double t=ms();f(x,w,al,o,Cin,H,W,Co);t=ms()-t;if(t<b)b=t;}return b;}
static void hdr(const char*first){printf("%-6s %-9s %9s",first,"Cin->Co","cur");for(int k=0;k<NC;++k)printf(" %8s",C[k].name);printf("   best\n");}

static void timing(void){
    printf("\n== timing at real shapes (min of %d reps, single thread) ==\n",REPS);
    printf("-- conv_real (3x3) --\n"); hdr("layer");
    {int Cin=CONV.Cin,H=CONV.H,W=CONV.W,Co=CONV.Cout;
     float*x=rf((size_t)Cin*H*W),*al=rf(Co);int8_t*w=rw((size_t)Co*Cin*9);float*o=malloc((size_t)Co*H*W*4);
     double base;{double b=1e30;for(int r=0;r<REPS;++r){double t=ms();conv_realin_naive(x,w,al,o,Cin,H,W,Co,3,3,1,1);t=ms()-t;if(t<b)b=t;}base=b;}
     printf("%-6s %3d->%-3d %9.1f",CONV.tag,Cin,Co,base);
     int bi=-1;double bm=1e30;
     for(int k=0;k<NC;++k){if(Co%C[k].cb){printf(" %8s","-");continue;}double t=t_conv(C[k].conv,x,w,al,o,Cin,H,W,Co);printf(" %8.1f",t);if(t<bm){bm=t;bi=k;}}
     printf("   %s (%.2fx)\n",bi>=0?C[bi].name:"?",base/bm);
     free(x);free(al);free(w);free(o);}
    printf("-- proj (1x1) --\n"); hdr("layer");
    double tot_cur=0,tot[NC];for(int k=0;k<NC;++k)tot[k]=0;double bestmix=0,tot_c8=0;const int c8=3;
    for(int s=0;s<NP;++s){int Cin=PROJ[s].Cin,H=PROJ[s].H,W=PROJ[s].W,Co=PROJ[s].Cout;
        float*x=rf((size_t)Cin*H*W),*Wm=rf((size_t)Co*Cin);float*o=malloc((size_t)Co*H*W*4);
        double base=t_proj(react_proj_1x1,x,Wm,o,Cin,H,W,Co);tot_cur+=base;
        printf("%-6s %3d->%-3d %9.1f",PROJ[s].tag,Cin,Co,base);
        int bi=-1;double bm=1e30;
        for(int k=0;k<NC;++k){if(Co%C[k].cb){printf(" %8s","-");continue;}double t=t_proj(C[k].proj,x,Wm,o,Cin,H,W,Co);tot[k]+=t;printf(" %8.1f",t);if(t<bm){bm=t;bi=k;}if(k==c8)tot_c8+=t;}
        bestmix+=bm;printf("   %s\n",bi>=0?C[bi].name:"?");
        free(x);free(Wm);free(o);}
    printf("%-6s %-9s %9.1f","TOTAL","",tot_cur);for(int k=0;k<NC;++k)printf(" %8.1f",tot[k]);printf("\n");
    printf("proj: current %.1f ms | c8p8 %.1f | best-per-shape %.1f ms\n",tot_cur,tot_c8,bestmix);
}

int main(int argc,char**argv){int ck=1,tm=1;if(argc>1&&!strcmp(argv[1],"check"))tm=0;if(argc>1&&!strcmp(argv[1],"time"))ck=0;if(ck&&check())return 1;if(tm)timing();return 0;}
