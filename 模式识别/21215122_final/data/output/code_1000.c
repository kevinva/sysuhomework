void qemu_set_fd_handler(int fd,

                         IOHandler *fd_read,

                         IOHandler *fd_write,

                         void *opaque)

{

    iohandler_init();

    aio_set_fd_handler(iohandler_ctx, fd, false,

                       fd_read, fd_write, NULL, opaque);

}
