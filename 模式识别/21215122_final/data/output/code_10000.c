yuv2mono_2_c_template(SwsContext *c, const uint16_t *buf0,

                      const uint16_t *buf1, const uint16_t *ubuf0,

                      const uint16_t *ubuf1, const uint16_t *vbuf0,

                      const uint16_t *vbuf1, const uint16_t *abuf0,

                      const uint16_t *abuf1, uint8_t *dest, int dstW,

                      int yalpha, int uvalpha, int y,

                      enum PixelFormat target)

{

    const uint8_t * const d128 = dither_8x8_220[y & 7];

    uint8_t *g = c->table_gU[128] + c->table_gV[128];

    int  yalpha1 = 4095 - yalpha;

    int i;



    for (i = 0; i < dstW - 7; i += 8) {

        int acc =    g[((buf0[i    ] * yalpha1 + buf1[i    ] * yalpha) >> 19) + d128[0]];

        acc += acc + g[((buf0[i + 1] * yalpha1 + buf1[i + 1] * yalpha) >> 19) + d128[1]];

        acc += acc + g[((buf0[i + 2] * yalpha1 + buf1[i + 2] * yalpha) >> 19) + d128[2]];

        acc += acc + g[((buf0[i + 3] * yalpha1 + buf1[i + 3] * yalpha) >> 19) + d128[3]];

        acc += acc + g[((buf0[i + 4] * yalpha1 + buf1[i + 4] * yalpha) >> 19) + d128[4]];

        acc += acc + g[((buf0[i + 5] * yalpha1 + buf1[i + 5] * yalpha) >> 19) + d128[5]];

        acc += acc + g[((buf0[i + 6] * yalpha1 + buf1[i + 6] * yalpha) >> 19) + d128[6]];

        acc += acc + g[((buf0[i + 7] * yalpha1 + buf1[i + 7] * yalpha) >> 19) + d128[7]];

        output_pixel(*dest++, acc);

    }

}
