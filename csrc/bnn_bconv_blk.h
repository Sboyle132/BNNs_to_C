/* bnn_bconv_blk.h - register-tiled binary-conv kernel TEMPLATE. Include with
 * CB, PB, SFX defined to emit one kernel bconv_blk<SFX>:
 *   CB  : output channels per register tile
 *   PB  : output pixels (along x) per register tile
 *   SFX : name suffix, e.g. _b8p2
 * A CB x PB tile of uint16x8 disagreement accumulators lives in registers
 * across the whole reduction; per tap-word each of PB activation vectors is
 * loaded once and reused across CB channels, each of CB weight vectors once and
 * reused across PB pixels. Keep CB*PB <= 16 (32 NEON regs).
 *
 * Requires the shared helpers from bnn_bconv_neon.c (pack_input,
 * pack_weights_ch, xor_popcnt, bconv_col_scalar, od). The interior fast path
 * assumes 3x3, pad 1, stride 1 (every react binary conv); any other geometry
 * falls back to all-scalar columns (correct, slow). Edges, the Cout%CB tail and
 * the pixel tail are handled by bconv_col_scalar. Scalar fallback path
 * (non-aarch64) computes the same integers. Validated bit-exactly vs
 * bconv_naive_int by bench_bconv. */
#ifndef BK_CAT
#define BK_CAT_(a, b) a##b
#define BK_CAT(a, b) BK_CAT_(a, b)
#endif
#define BK_NAME BK_CAT(bconv_blk, SFX)

void BK_NAME(const int8_t *a, const uint64_t *packed_w, int32_t *P,
             int Cin, int H, int W, int Cout, int kh, int kw, int pad, int stride) {
    int Hout = od(H, kh, pad, stride), Wout = od(W, kw, pad, stride);
    int Wc = (Cin + 63) / 64;
    uint64_t *apack = pack_input(a, Cin, H, W, Wc);
    uint64_t *wpack = pack_weights_ch(packed_w, Cout, Cin, kh, kw, Wc);
    int general = !(kh == 3 && kw == 3 && pad == 1 && stride == 1);
    int coCB = Cout - Cout % CB;

    for (int oy = 0; oy < Hout; ++oy) {
        int oyb = oy * stride - pad;
        int ky0 = oyb < 0 ? -oyb : 0, ky1 = (oyb + kh > H) ? H - oyb : kh;
        if (general) {
            for (int ox = 0; ox < Wout; ++ox)
                bconv_col_scalar(apack, wpack, Wc, Cin, kh, kw, H, W, Hout, Wout,
                                 oy, oyb, ky0, ky1, ox, stride, pad, P, 0, Cout);
            continue;
        }
        for (int co = 0; co < coCB; co += CB) {
            const uint64_t *wc[CB];
            for (int c = 0; c < CB; ++c) wc[c] = wpack + (size_t)((co + c) * 3) * 3 * Wc;
            int ox = 0;
            while (ox < Wout) {
                if (!(ox >= 1 && ox + PB <= Wout - 1)) {       /* edge column: scalar */
                    bconv_col_scalar(apack, wpack, Wc, Cin, 3, 3, H, W, Hout, Wout,
                                     oy, oyb, ky0, ky1, ox, 1, 1, P, co, co + CB);
                    ox += 1;
                    continue;
                }
                /* interior tile: PB pixels [ox,ox+PB), CB channels, full kx */
                uint32_t dsc[CB][PB];
                for (int c = 0; c < CB; ++c) for (int j = 0; j < PB; ++j) dsc[c][j] = 0;
#ifdef __aarch64__
                uint16x8_t acc[CB][PB];
                for (int c = 0; c < CB; ++c) for (int j = 0; j < PB; ++j) acc[c][j] = vdupq_n_u16(0);
#endif
                for (int ky = ky0; ky < ky1; ++ky) {
                    int iy = oyb + ky;
                    for (int kx = 0; kx < 3; ++kx) {
                        const uint64_t *abase = apack + (size_t)(iy * W + (ox - 1) + kx) * Wc;
                        int tap = ky * 3 + kx;
                        int w = 0;
#ifdef __aarch64__
                        for (; w + 2 <= Wc; w += 2) {
                            uint8x16_t av[PB];
                            for (int j = 0; j < PB; ++j) av[j] = vld1q_u8((const uint8_t *)(abase + (size_t)j * Wc + w));
                            for (int c = 0; c < CB; ++c) {
                                uint8x16_t wv = vld1q_u8((const uint8_t *)(wc[c] + (size_t)tap * Wc + w));
                                for (int j = 0; j < PB; ++j)
                                    acc[c][j] = vpadalq_u8(acc[c][j], vcntq_u8(veorq_u8(av[j], wv)));
                            }
                        }
#endif
                        for (; w < Wc; ++w) {                  /* odd tail word / scalar fallback */
                            uint64_t av[PB];
                            for (int j = 0; j < PB; ++j) av[j] = abase[(size_t)j * Wc + w];
                            for (int c = 0; c < CB; ++c) {
                                uint64_t wv = wc[c][(size_t)tap * Wc + w];
                                for (int j = 0; j < PB; ++j) dsc[c][j] += __builtin_popcountll(av[j] ^ wv);
                            }
                        }
                    }
                }
                int vt = (ky1 - ky0) * 3 * Cin;
                for (int c = 0; c < CB; ++c)
                    for (int j = 0; j < PB; ++j) {
                        uint32_t D = dsc[c][j];
#ifdef __aarch64__
                        D += vaddvq_u32(vpaddlq_u16(acc[c][j]));
#endif
                        P[((size_t)(co + c) * Hout + oy) * Wout + (ox + j)] = vt - 2 * (int)D;
                    }
                ox += PB;
            }
        }
        for (int co = coCB; co < Cout; ++co)                   /* channel tail */
            for (int ox = 0; ox < Wout; ++ox)
                bconv_col_scalar(apack, wpack, Wc, Cin, 3, 3, H, W, Hout, Wout,
                                 oy, oyb, ky0, ky1, ox, 1, 1, P, co, co + 1);
    }
    rs_free(apack); rs_free(wpack);
}

#undef BK_NAME
#undef CB
#undef PB
#undef SFX
