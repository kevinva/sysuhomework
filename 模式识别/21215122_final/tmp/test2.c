int bdrv_password_cb(void *mon, const char *password, void *opaque)
{
    void *bs = opaque;
    int ret = 0;
    if (bdrv_set_key(bs, password) != 0) {
        monitor_printf(mon, "invalid password\n");
        ret = -EPERM;
    }
    
    if (mon->password_completion_cb)
        mon->password_completion_cb(mon->password_opaque, ret);
    
    monitor_read_command(mon, 1);
}

// 将自定义指针赋值为void *