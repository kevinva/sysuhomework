static int tap_alloc(char *dev, size_t dev_size)

{

    int tap_fd, if_fd, ppa = -1;

    static int ip_fd = 0;

    char *ptr;



    static int arp_fd = 0;

    int ip_muxid, arp_muxid;

    struct strioctl  strioc_if, strioc_ppa;

    int link_type = I_PLINK;

    struct lifreq ifr;

    char actual_name[32] = "";



    memset(&ifr, 0x0, sizeof(ifr));



    if( *dev ){

       ptr = dev;

       while( *ptr && !qemu_isdigit((int)*ptr) ) ptr++;

       ppa = atoi(ptr);

    }



    /* Check if IP device was opened */

    if( ip_fd )

       close(ip_fd);



    TFR(ip_fd = open("/dev/udp", O_RDWR, 0));

    if (ip_fd < 0) {

       syslog(LOG_ERR, "Can't open /dev/ip (actually /dev/udp)");

       return -1;

    }



    TFR(tap_fd = open("/dev/tap", O_RDWR, 0));

    if (tap_fd < 0) {

       syslog(LOG_ERR, "Can't open /dev/tap");

       return -1;

    }



    /* Assign a new PPA and get its unit number. */

    strioc_ppa.ic_cmd = TUNNEWPPA;

    strioc_ppa.ic_timout = 0;

    strioc_ppa.ic_len = sizeof(ppa);

    strioc_ppa.ic_dp = (char *)&ppa;

    if ((ppa = ioctl (tap_fd, I_STR, &strioc_ppa)) < 0)

       syslog (LOG_ERR, "Can't assign new interface");



    TFR(if_fd = open("/dev/tap", O_RDWR, 0));

    if (if_fd < 0) {

       syslog(LOG_ERR, "Can't open /dev/tap (2)");

       return -1;

    }

    if(ioctl(if_fd, I_PUSH, "ip") < 0){

       syslog(LOG_ERR, "Can't push IP module");

       return -1;

    }



    if (ioctl(if_fd, SIOCGLIFFLAGS, &ifr) < 0)

	syslog(LOG_ERR, "Can't get flags\n");



    snprintf (actual_name, 32, "tap%d", ppa);

    pstrcpy(ifr.lifr_name, sizeof(ifr.lifr_name), actual_name);



    ifr.lifr_ppa = ppa;

    /* Assign ppa according to the unit number returned by tun device */



    if (ioctl (if_fd, SIOCSLIFNAME, &ifr) < 0)

        syslog (LOG_ERR, "Can't set PPA %d", ppa);

    if (ioctl(if_fd, SIOCGLIFFLAGS, &ifr) <0)

        syslog (LOG_ERR, "Can't get flags\n");

    /* Push arp module to if_fd */

    if (ioctl (if_fd, I_PUSH, "arp") < 0)

        syslog (LOG_ERR, "Can't push ARP module (2)");



    /* Push arp module to ip_fd */

    if (ioctl (ip_fd, I_POP, NULL) < 0)

        syslog (LOG_ERR, "I_POP failed\n");

    if (ioctl (ip_fd, I_PUSH, "arp") < 0)

        syslog (LOG_ERR, "Can't push ARP module (3)\n");

    /* Open arp_fd */

    TFR(arp_fd = open ("/dev/tap", O_RDWR, 0));

    if (arp_fd < 0)

       syslog (LOG_ERR, "Can't open %s\n", "/dev/tap");



    /* Set ifname to arp */

    strioc_if.ic_cmd = SIOCSLIFNAME;

    strioc_if.ic_timout = 0;

    strioc_if.ic_len = sizeof(ifr);

    strioc_if.ic_dp = (char *)&ifr;

    if (ioctl(arp_fd, I_STR, &strioc_if) < 0){

        syslog (LOG_ERR, "Can't set ifname to arp\n");

    }



    if((ip_muxid = ioctl(ip_fd, I_LINK, if_fd)) < 0){

       syslog(LOG_ERR, "Can't link TAP device to IP");

       return -1;

    }



    if ((arp_muxid = ioctl (ip_fd, link_type, arp_fd)) < 0)

        syslog (LOG_ERR, "Can't link TAP device to ARP");



    close (if_fd);



    memset(&ifr, 0x0, sizeof(ifr));

    pstrcpy(ifr.lifr_name, sizeof(ifr.lifr_name), actual_name);

    ifr.lifr_ip_muxid  = ip_muxid;

    ifr.lifr_arp_muxid = arp_muxid;



    if (ioctl (ip_fd, SIOCSLIFMUXID, &ifr) < 0)

    {

      ioctl (ip_fd, I_PUNLINK , arp_muxid);

      ioctl (ip_fd, I_PUNLINK, ip_muxid);

      syslog (LOG_ERR, "Can't set multiplexor id");

    }



    snprintf(dev, dev_size, "tap%d", ppa);

    return tap_fd;

}
