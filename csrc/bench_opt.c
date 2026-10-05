/* bench_opt.c - three independent candidates, side by side vs current, at real
 * shapes. head / maxpool(real+int) / tap->bitpack fusion. Each row is its own
 * merge decision. Correctness is exact where integer (maxpool, pack), tol where
 * float (head). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

/* current kernels */
void head_1x1(const float *, const float *, const float *, float *, int, int, int, int);
void react_tap(const float *, int8_t *, int, int, int, const float *);
/* candidates */
void head_1x1_v2(const float *, const float *, const float *, float *, int, int, int, int);
void maxpool_real_v2(const float *, float *, int, int, int, int, int);
void maxpool_P_v2(const int32_t *, int32_t *, int, int, int, int, int);
void tap_pack(const float *, uint64_t *, int, int, int, const float *);

/* local references */
static void maxpool_real_ref(const float *x, float *o, int C, int H, int W) {
    int Ho=(H-2)/2+1, Wo=(W-2)/2+1;
    for(int c=0;c<C;++c)for(int oy=0;oy<Ho;++oy)for(int ox=0;ox<Wo;++ox){
        const float*xc=x+(long)c*H*W; float m=xc[(2*oy)*W+2*ox];
        for(int a=0;a<2;++a)for(int b=0;b<2;++b){float v=xc[(2*oy+a)*W+2*ox+b];if(v>m)m=v;}
        o[(long)c*Ho*Wo+oy*Wo+ox]=m;}
}
static void maxpool_P_ref(const int32_t *x, int32_t *o, int C, int H, int W) {
    int Ho=(H-2)/2+1, Wo=(W-2)/2+1;
    for(int c=0;c<C;++c)for(int oy=0;oy<Ho;++oy)for(int ox=0;ox<Wo;++ox){
        const int32_t*xc=x+(long)c*H*W; int32_t m=xc[(2*oy)*W+2*ox];
        for(int a=0;a<2;++a)for(int b=0;b<2;++b){int32_t v=xc[(2*oy+a)*W+2*ox+b];if(v>m)m=v;}
        o[(long)c*Ho*Wo+oy*Wo+ox]=m;}
}
static void pack_input_ref(const int8_t *a, int Cin, int H, int W, int Wc, uint64_t *apack) {
    memset(apack,0,(size_t)H*W*Wc*8);
    for(int ci=0;ci<Cin;++ci){int wo=ci>>6;uint64_t bit=(uint64_t)1<<(ci&63);const int8_t*ac=a+(size_t)ci*H*W;
        for(int p=0;p<H*W;++p) if(ac[p]>0) apack[(size_t)p*Wc+wo]|=bit;}
}

static uint64_t rs=0xABCDEF;
static float rf1(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return (float)((rs>>40)/8388608.0-1.0);}
static float*rf(size_t n){float*p=malloc(n*4);for(size_t i=0;i<n;++i)p[i]=rf1();return p;}
static int32_t*ri(size_t n){int32_t*p=malloc(n*4);for(size_t i=0;i<n;++i)p[i]=(int32_t)(rf1()*50);return p;}
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static double reld(const float*a,const float*b,size_t n){double d=0,m=1e-30;for(size_t i=0;i<n;++i){double e=fabs((double)a[i]-b[i]);if(e>d)d=e;double r=fabs((double)b[i]);if(r>m)m=r;}return d/m;}
#define R 5
#define BEST(call) ({double b=1e30;for(int r=0;r<R;++r){double t=ms();call;t=ms()-t;if(t<b)b=t;}b;})

typedef struct{const char*tag;int C,H,W,Co,n;} Sh;

int main(void){
    int bad=0;
    /* --- HEAD: only enc1 head, 16->3 @ full res --- */
    printf("== HEAD (1x1 16->3) ==\n");
    {int Cin=16,H=598,W=1092,Co=3;size_t n=(size_t)Cin*H*W,no=(size_t)Co*H*W;
     float*a=rf(n),*Wm=rf((size_t)Co*Cin),*bi=rf(Co),*o1=malloc(no*4),*o2=malloc(no*4);
     head_1x1(a,Wm,bi,o1,Cin,H,W,Co); head_1x1_v2(a,Wm,bi,o2,Cin,H,W,Co);
     double d=reld(o2,o1,no); if(d>1e-4)bad++;
     double tc=BEST(head_1x1(a,Wm,bi,o1,Cin,H,W,Co)), tn=BEST(head_1x1_v2(a,Wm,bi,o2,Cin,H,W,Co));
     printf("  current %.2f ms -> v2 %.2f ms  (%.2fx)  reldiff %.0e\n",tc,tn,tc/tn,d);
     free(a);free(Wm);free(bi);free(o1);free(o2);}

    /* --- MAXPOOL: the 8 pool calls (4 int from conv2 P, 4 real from h1) --- */
    printf("== MAXPOOL (2x2) ==  [int P and real h1, down branches of enc1-4]\n");
    Sh pool[]={{"enc1",16,598,1092,0,1},{"enc2",32,299,546,0,1},{"enc3",64,149,273,0,1},{"enc4",128,74,136,0,1}};
    {double rc_t=0,rn_t=0,ic_t=0,in_t=0;
     for(int s=0;s<4;++s){int C=pool[s].C,H=pool[s].H,W=pool[s].W;size_t n=(size_t)C*H*W;
        int Ho=(H-2)/2+1,Wo=(W-2)/2+1;size_t no=(size_t)C*Ho*Wo;
        float*x=rf(n),*o1=malloc(no*4),*o2=malloc(no*4);int32_t*xi=ri(n),*i1=malloc(no*4),*i2=malloc(no*4);
        maxpool_real_ref(x,o1,C,H,W); maxpool_real_v2(x,o2,C,H,W,2,2);
        if(memcmp(o1,o2,no*4)){ if(reld(o2,o1,no)>0)bad++; }
        maxpool_P_ref(xi,i1,C,H,W); maxpool_P_v2(xi,i2,C,H,W,2,2);
        if(memcmp(i1,i2,no*4))bad++;
        rc_t+=BEST(maxpool_real_ref(x,o1,C,H,W)); rn_t+=BEST(maxpool_real_v2(x,o2,C,H,W,2,2));
        ic_t+=BEST(maxpool_P_ref(xi,i1,C,H,W)); in_t+=BEST(maxpool_P_v2(xi,i2,C,H,W,2,2));
        free(x);free(o1);free(o2);free(xi);free(i1);free(i2);}
     printf("  real: scalar %.2f ms -> v2 %.2f ms (%.2fx)\n",rc_t,rn_t,rc_t/rn_t);
     printf("  int : scalar %.2f ms -> v2 %.2f ms (%.2fx)\n",ic_t,in_t,ic_t/in_t);}

    /* --- TAP->BITPACK FUSION: all 17 bconv inputs. current = react_tap(int8)+pack_input;
           fused = tap_pack. Verify apack identical. --- */
    printf("== TAP->BITPACK (17 bconv inputs) ==  current=tap+pack  vs  fused=tap_pack\n");
    Sh bc[]={{"enc1.u2",16,598,1092,16,1},{"enc2.u1",16,299,546,32,1},{"enc2.u2",32,299,546,32,1},
        {"enc3.u1",32,149,273,64,1},{"enc3.u2",64,149,273,64,1},{"enc4.u1",64,74,136,128,1},
        {"enc4.u2",128,74,136,128,1},{"bott.u1",128,37,68,256,1},{"bott.u2",256,37,68,256,1},
        {"dec4.u1",384,74,136,128,1},{"dec4.u2",128,74,136,128,1},{"dec3.u1",192,149,273,64,1},
        {"dec3.u2",64,149,273,64,1},{"dec2.u1",96,299,546,32,1},{"dec2.u2",32,299,546,32,1},
        {"dec1.u1",48,598,1092,16,1},{"dec1.u2",16,598,1092,16,1}};
    {double cur=0,fus=0;
     for(int s=0;s<17;++s){int C=bc[s].C,H=bc[s].H,W=bc[s].W,Wc=(C+63)/64;size_t n=(size_t)C*H*W;
        float*x=rf(n),*rsgn=rf(C); int8_t*a8=malloc(n); uint64_t*ap1=malloc((size_t)H*W*Wc*8),*ap2=malloc((size_t)H*W*Wc*8);
        react_tap(x,a8,C,H,W,rsgn); pack_input_ref(a8,C,H,W,Wc,ap1);
        tap_pack(x,ap2,C,H,W,rsgn);
        if(memcmp(ap1,ap2,(size_t)H*W*Wc*8)){bad++;printf("  %s APACK MISMATCH\n",bc[s].tag);}
        double tc=BEST((react_tap(x,a8,C,H,W,rsgn),pack_input_ref(a8,C,H,W,Wc,ap1)));
        double tf=BEST(tap_pack(x,ap2,C,H,W,rsgn));
        cur+=tc; fus+=tf;
        free(x);free(rsgn);free(a8);free(ap1);free(ap2);}
     printf("  TOTAL: tap+pack %.1f ms -> tap_pack %.1f ms  (%.2fx, saves %.1f ms)\n",cur,fus,cur/fus,cur-fus);}

    printf("\n%s\n", bad?"CORRECTNESS: FAIL":"CORRECTNESS: PASS");
    return bad?1:0;
}
