static av_cold int fdk_aac_decode_init(AVCodecContext *avctx)

{

    FDKAACDecContext *s = avctx->priv_data;

    AAC_DECODER_ERROR err;

    int ret;



    s->handle = aacDecoder_Open(avctx->extradata_size ? TT_MP4_RAW : TT_MP4_ADTS, 1);

    if (!s->handle) {

        av_log(avctx, AV_LOG_ERROR, "Error opening decoder\n");

        return AVERROR_UNKNOWN;

    }



    if (avctx->extradata_size) {

        if ((err = aacDecoder_ConfigRaw(s->handle, &avctx->extradata,

                                        &avctx->extradata_size)) != AAC_DEC_OK) {

            av_log(avctx, AV_LOG_ERROR, "Unable to set extradata\n");

            return AVERROR_INVALIDDATA;

        }

    }



    if ((err = aacDecoder_SetParam(s->handle, AAC_CONCEAL_METHOD,

                                   s->conceal_method)) != AAC_DEC_OK) {

        av_log(avctx, AV_LOG_ERROR, "Unable to set error concealment method\n");

        return AVERROR_UNKNOWN;

    }



    if (avctx->request_channel_layout > 0 &&

        avctx->request_channel_layout != AV_CH_LAYOUT_NATIVE) {

        int downmix_channels = -1;



        switch (avctx->request_channel_layout) {

        case AV_CH_LAYOUT_STEREO:

        case AV_CH_LAYOUT_STEREO_DOWNMIX:

            downmix_channels = 2;

            break;

        case AV_CH_LAYOUT_MONO:

            downmix_channels = 1;

            break;

        default:

            av_log(avctx, AV_LOG_WARNING, "Invalid request_channel_layout\n");

            break;

        }



        if (downmix_channels != -1) {

            if (aacDecoder_SetParam(s->handle, AAC_PCM_MAX_OUTPUT_CHANNELS,

                                    downmix_channels) != AAC_DEC_OK) {

               av_log(avctx, AV_LOG_WARNING, "Unable to set output channels in the decoder\n");

            } else {

               s->anc_buffer = av_malloc(DMX_ANC_BUFFSIZE);

               if (!s->anc_buffer) {

                   av_log(avctx, AV_LOG_ERROR, "Unable to allocate ancillary buffer for the decoder\n");

                   ret = AVERROR(ENOMEM);

                   goto fail;

               }

               if (aacDecoder_AncDataInit(s->handle, s->anc_buffer, DMX_ANC_BUFFSIZE)) {

                   av_log(avctx, AV_LOG_ERROR, "Unable to register downmix ancillary buffer in the decoder\n");

                   ret = AVERROR_UNKNOWN;

                   goto fail;

               }

            }

        }

    }



    if (s->drc_boost != -1) {

        if (aacDecoder_SetParam(s->handle, AAC_DRC_BOOST_FACTOR, s->drc_boost) != AAC_DEC_OK) {

            av_log(avctx, AV_LOG_ERROR, "Unable to set DRC boost factor in the decoder\n");

            return AVERROR_UNKNOWN;

        }

    }



    if (s->drc_cut != -1) {

        if (aacDecoder_SetParam(s->handle, AAC_DRC_ATTENUATION_FACTOR, s->drc_cut) != AAC_DEC_OK) {

            av_log(avctx, AV_LOG_ERROR, "Unable to set DRC attenuation factor in the decoder\n");

            return AVERROR_UNKNOWN;

        }

    }



    if (s->drc_level != -1) {

        if (aacDecoder_SetParam(s->handle, AAC_DRC_REFERENCE_LEVEL, s->drc_level) != AAC_DEC_OK) {

            av_log(avctx, AV_LOG_ERROR, "Unable to set DRC reference level in the decoder\n");

            return AVERROR_UNKNOWN;

        }

    }



    if (s->drc_heavy != -1) {

        if (aacDecoder_SetParam(s->handle, AAC_DRC_HEAVY_COMPRESSION, s->drc_heavy) != AAC_DEC_OK) {

            av_log(avctx, AV_LOG_ERROR, "Unable to set DRC heavy compression in the decoder\n");

            return AVERROR_UNKNOWN;

        }

    }



#ifdef AACDECODER_LIB_VL0

    if (aacDecoder_SetParam(s->handle, AAC_PCM_LIMITER_ENABLE, s->level_limit) != AAC_DEC_OK) {

        av_log(avctx, AV_LOG_ERROR, "Unable to set in signal level limiting in the decoder\n");

        return AVERROR_UNKNOWN;

    }

#endif



    avctx->sample_fmt = AV_SAMPLE_FMT_S16;



    s->decoder_buffer_size = DECODER_BUFFSIZE * DECODER_MAX_CHANNELS;

    s->decoder_buffer = av_malloc(s->decoder_buffer_size);

    if (!s->decoder_buffer) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    return 0;

fail:

    fdk_aac_decode_close(avctx);

    return ret;

}
