static av_cold int ffat_close_encoder(AVCodecContext *avctx)

{

    ATDecodeContext *at = avctx->priv_data;

    AudioConverterDispose(at->converter);

    av_frame_unref(&at->new_in_frame);

    av_frame_unref(&at->in_frame);

    ff_af_queue_close(&at->afq);

    return 0;

}
