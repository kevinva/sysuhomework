static int32_t virtio_net_flush_tx(VirtIONet *n, VirtQueue *vq)

{

    VirtQueueElement elem;

    int32_t num_packets = 0;



    if (!(n->vdev.status & VIRTIO_CONFIG_S_DRIVER_OK)) {

        return num_packets;

    }



    if (n->async_tx.elem.out_num) {

        virtio_queue_set_notification(n->tx_vq, 0);

        return num_packets;

    }



    while (virtqueue_pop(vq, &elem)) {

        ssize_t ret, len = 0;

        unsigned int out_num = elem.out_num;

        struct iovec *out_sg = &elem.out_sg[0];

        unsigned hdr_len;



        /* hdr_len refers to the header received from the guest */

        hdr_len = n->mergeable_rx_bufs ?

            sizeof(struct virtio_net_hdr_mrg_rxbuf) :

            sizeof(struct virtio_net_hdr);



        if (out_num < 1 || out_sg->iov_len != hdr_len) {

            error_report("virtio-net header not in first element");

            exit(1);

        }



        /* ignore the header if GSO is not supported */

        if (!n->has_vnet_hdr) {

            out_num--;

            out_sg++;

            len += hdr_len;

        } else if (n->mergeable_rx_bufs) {

            /* tapfd expects a struct virtio_net_hdr */

            hdr_len -= sizeof(struct virtio_net_hdr);

            out_sg->iov_len -= hdr_len;

            len += hdr_len;

        }



        ret = qemu_sendv_packet_async(&n->nic->nc, out_sg, out_num,

                                      virtio_net_tx_complete);

        if (ret == 0) {

            virtio_queue_set_notification(n->tx_vq, 0);

            n->async_tx.elem = elem;

            n->async_tx.len  = len;

            return -EBUSY;

        }



        len += ret;



        virtqueue_push(vq, &elem, len);

        virtio_notify(&n->vdev, vq);



        if (++num_packets >= n->tx_burst) {

            break;

        }

    }

    return num_packets;

}
