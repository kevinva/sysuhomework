static int parse_tag(AVFormatContext *s, const uint8_t *buf)

{

    int genre;



    if (!(buf[0] == 'T' &&

          buf[1] == 'A' &&

          buf[2] == 'G'))

        return -1;

    get_string(s, "title",   buf +  3, 30);

    get_string(s, "artist",  buf + 33, 30);

    get_string(s, "album",   buf + 63, 30);

    get_string(s, "date",    buf + 93,  4);

    get_string(s, "comment", buf + 97, 30);

    if (buf[125] == 0 && buf[126] != 0)

        av_metadata_set2(&s->metadata, "track", av_d2str(buf[126]), AV_METADATA_DONT_STRDUP_VAL);

    genre = buf[127];

    if (genre <= ID3v1_GENRE_MAX)

        av_metadata_set2(&s->metadata, "genre", ff_id3v1_genre_str[genre], 0);

    return 0;

}
