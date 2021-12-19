static void qemu_laio_process_completion(struct qemu_laiocb *laiocb)

{

    int ret;



    ret = laiocb->ret;

    if (ret != -ECANCELED) {

        if (ret == laiocb->nbytes) {

            ret = 0;

        } else if (ret >= 0) {

            /* Short reads mean EOF, pad with zeros. */

            if (laiocb->is_read) {

                qemu_iovec_memset(laiocb->qiov, ret, 0,

                    laiocb->qiov->size - ret);

            } else {

                ret = -ENOSPC;

            }

        }

    }



    laiocb->ret = ret;

    if (laiocb->co) {

        qemu_coroutine_enter(laiocb->co);

    } else {

        laiocb->common.cb(laiocb->common.opaque, ret);

        qemu_aio_unref(laiocb);

    }

}
