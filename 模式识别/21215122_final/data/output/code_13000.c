static int ptx_decode_frame(AVCodecContext *avctx, void *data, int *data_size,
                            AVPacket *avpkt) {
    const uint8_t *buf = avpkt->data;
    const uint8_t *buf_end = avpkt->data + avpkt->size;
    PTXContext * const s = avctx->priv_data;
    AVFrame *picture = data;
    AVFrame * const p = &s->picture;
    unsigned int offset, w, h, y, stride, bytes_per_pixel;
    uint8_t *ptr;
    if (buf_end - buf < 14)
    offset          = AV_RL16(buf);
    w               = AV_RL16(buf+8);
    h               = AV_RL16(buf+10);
    bytes_per_pixel = AV_RL16(buf+12) >> 3;
    if (bytes_per_pixel != 2) {
        av_log_ask_for_sample(avctx, "Image format is not RGB15.\n");
        return -1;
    }
    avctx->pix_fmt = PIX_FMT_RGB555;
    if (offset != 0x2c)
        av_log_ask_for_sample(avctx, "offset != 0x2c\n");
    buf += offset;
    if (p->data[0])
        avctx->release_buffer(avctx, p);
    if (av_image_check_size(w, h, 0, avctx))
        return -1;
    if (w != avctx->width || h != avctx->height)
        avcodec_set_dimensions(avctx, w, h);
    if (avctx->get_buffer(avctx, p) < 0) {
        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");
        return -1;
    }
    p->pict_type = AV_PICTURE_TYPE_I;
    ptr    = p->data[0];
    stride = p->linesize[0];
    for (y=0; y<h; y++) {
        if (buf_end - buf < w * bytes_per_pixel)
            break;
#if HAVE_BIGENDIAN
        unsigned int x;
        for (x=0; x<w*bytes_per_pixel; x+=bytes_per_pixel)
            AV_WN16(ptr+x, AV_RL16(buf+x));
#else
        memcpy(ptr, buf, w*bytes_per_pixel);
#endif
        ptr += stride;
        buf += w*bytes_per_pixel;
    }
    *picture = s->picture;
    *data_size = sizeof(AVPicture);
    return offset + w*h*bytes_per_pixel;
}