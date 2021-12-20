static void pxa2xx_fir_write(void *opaque, hwaddr addr,

                             uint64_t value64, unsigned size)

{

    PXA2xxFIrState *s = (PXA2xxFIrState *) opaque;

    uint32_t value = value64;

    uint8_t ch;



    switch (addr) {

    case ICCR0:

        s->control[0] = value;

        if (!(value & (1 << 4)))			/* RXE */

            s->rx_len = s->rx_start = 0;

        if (!(value & (1 << 3))) {                      /* TXE */

            /* Nop */

        }

        s->enable = value & 1;				/* ITR */

        if (!s->enable)

            s->status[0] = 0;

        pxa2xx_fir_update(s);

        break;

    case ICCR1:

        s->control[1] = value;

        break;

    case ICCR2:

        s->control[2] = value & 0x3f;

        pxa2xx_fir_update(s);

        break;

    case ICDR:

        if (s->control[2] & (1 << 2)) { /* TXP */

            ch = value;

        } else {

            ch = ~value;

        }

        if (s->enable && (s->control[0] & (1 << 3))) { /* TXE */

            /* XXX this blocks entire thread. Rewrite to use

             * qemu_chr_fe_write and background I/O callbacks */

            qemu_chr_fe_write_all(&s->chr, &ch, 1);

        }

        break;

    case ICSR0:

        s->status[0] &= ~(value & 0x66);

        pxa2xx_fir_update(s);

        break;

    case ICFOR:

        break;

    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

    }

}
