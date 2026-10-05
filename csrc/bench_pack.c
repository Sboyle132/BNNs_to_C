/* bench_pack.c - track 3 go/no-go: measure what the per-call input bitpacking
 * (pack_input, done inside every bconv) costs at the 17 binary-conv shapes.
 * This is the prize for the tap->bitpack fusion (daBNN/LCE "bitpacked output"):
 * if the tap wrote channel-packed bits directly, this pass plus the int8 tap
 * write both disappear. Compares against the current bconv total (~4.9s, b4p4).
 * Local copy of pack_input identical to bnn_bconv_neon.c's. */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static void pack_input(const int8_t *a, int Cin, int H, int W, int Wc, uint64_t *apack) {
    memset(apack, 0, (size_t)H * W * Wc * sizeof(uint64_t));
    for (int ci = 0; ci < Cin; ++ci) {
        int wo = ci >> 6; uint64_t bit = (uint64_t)1 << (ci & 63);
        const int8_t *ac = a + (size_t)ci * H * W;
        for (int p = 0; p < H * W; ++p)
            if (ac[p] > 0) apack[(size_t)p * Wc + wo] |= bit;
    }
}

typedef struct { const char *tag; int Cin, H, W, Cout; } Shape;
static const Shape SH[] = {
    {"enc1.u2",16,598,1092,16},{"enc2.u1",16,299,546,32},{"enc2.u2",32,299,546,32},
    {"enc3.u1",32,149,273,64},{"enc3.u2",64,149,273,64},{"enc4.u1",64,74,136,128},
    {"enc4.u2",128,74,136,128},{"bott.u1",128,37,68,256},{"bott.u2",256,37,68,256},
    {"dec4.u1",384,74,136,128},{"dec4.u2",128,74,136,128},{"dec3.u1",192,149,273,64},
    {"dec3.u2",64,149,273,64},{"dec2.u1",96,299,546,32},{"dec2.u2",32,299,546,32},
    {"dec1.u1",48,598,1092,16},{"dec1.u2",16,598,1092,16},
};
#define NSH (int)(sizeof SH/sizeof SH[0])
static uint64_t rs=0x12345;
static int8_t rpm(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return (rs>>40)&1?1:-1;}
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}

int main(void){
    double tot=0;
    printf("== input bitpack cost per bconv call (min of 5) ==\n");
    printf("%-9s %-9s %4s %10s\n","layer","Cin->Co","Wc","pack_ms");
    for(int s=0;s<NSH;++s){
        int Cin=SH[s].Cin,H=SH[s].H,W=SH[s].W,Wc=(Cin+63)/64;
        int8_t*a=malloc((size_t)Cin*H*W);for(size_t i=0;i<(size_t)Cin*H*W;++i)a[i]=rpm();
        uint64_t*ap=malloc((size_t)H*W*Wc*8);
        double b=1e30;for(int r=0;r<5;++r){double t=ms();pack_input(a,Cin,H,W,Wc,ap);t=ms()-t;if(t<b)b=t;}
        printf("%-9s %3d->%-3d %4d %10.2f\n",SH[s].tag,Cin,SH[s].Cout,Wc,b);
        tot+=b;free(a);free(ap);
    }
    printf("TOTAL input-pack: %.1f ms  (vs bconv ~4900 ms b4p4 -> %.1f%% of bconv)\n",tot,100*tot/4900.0);
    return 0;
}
