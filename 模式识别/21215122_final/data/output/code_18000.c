static int spapr_vty_init(VIOsPAPRDevice *sdev)
{
    VIOsPAPRVTYDevice *dev = (VIOsPAPRVTYDevice *)sdev;
    qemu_chr_add_handlers(dev->chardev, vty_can_receive,
                          vty_receive, NULL, dev);
    return 0;