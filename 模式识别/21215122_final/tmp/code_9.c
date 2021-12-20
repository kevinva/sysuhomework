static void exynos4210_mct_write(void *opaque, target_phys_addr_t offset,

        uint64_t value, unsigned size)

{

    Exynos4210MCTState *s = (Exynos4210MCTState *)opaque;

    int index;  /* index in buffer which represents register set */

    int shift;

    int lt_i;

    uint64_t new_frc;

    uint32_t i;

    uint32_t old_val;

#ifdef DEBUG_MCT

    static uint32_t icntb_max[2] = {0};

    static uint32_t icntb_min[2] = {UINT32_MAX, UINT32_MAX};

    static uint32_t tcntb_max[2] = {0};

    static uint32_t tcntb_min[2] = {UINT32_MAX, UINT32_MAX};

#endif



    new_frc = s->g_timer.reg.cnt;



    switch (offset) {



    case MCT_CFG:

        s->reg_mct_cfg = value;

        exynos4210_mct_update_freq(s);

        break;



    case G_CNT_L:

    case G_CNT_U:

        if (offset == G_CNT_L) {



            DPRINTF("global timer write to reg.cntl %llx\n", value);



            new_frc = (s->g_timer.reg.cnt & (uint64_t)UINT32_MAX << 32) + value;

            s->g_timer.reg.cnt_wstat |= G_CNT_WSTAT_L;

        }

        if (offset == G_CNT_U) {



            DPRINTF("global timer write to reg.cntu %llx\n", value);



            new_frc = (s->g_timer.reg.cnt & UINT32_MAX) +

                    ((uint64_t)value << 32);

            s->g_timer.reg.cnt_wstat |= G_CNT_WSTAT_U;

        }



        s->g_timer.reg.cnt = new_frc;

        exynos4210_gfrc_restart(s);

        break;



    case G_CNT_WSTAT:

        s->g_timer.reg.cnt_wstat &= ~(value);

        break;



    case G_COMP_L(0): case G_COMP_L(1): case G_COMP_L(2): case G_COMP_L(3):

    case G_COMP_U(0): case G_COMP_U(1): case G_COMP_U(2): case G_COMP_U(3):

    index = GET_G_COMP_IDX(offset);

    shift = 8 * (offset & 0x4);

    s->g_timer.reg.comp[index] =

            (s->g_timer.reg.comp[index] &

            (((uint64_t)UINT32_MAX << 32) >> shift)) +

            (value << shift);



    DPRINTF("comparator %d write 0x%llx val << %d\n", index, value, shift);



    if (offset&0x4) {

        s->g_timer.reg.wstat |= G_WSTAT_COMP_U(index);

    } else {

        s->g_timer.reg.wstat |= G_WSTAT_COMP_L(index);

    }



    exynos4210_gfrc_restart(s);

    break;



    case G_TCON:

        old_val = s->g_timer.reg.tcon;

        s->g_timer.reg.tcon = value;

        s->g_timer.reg.wstat |= G_WSTAT_TCON_WRITE;



        DPRINTF("global timer write to reg.g_tcon %llx\n", value);



        /* Start FRC if transition from disabled to enabled */

        if ((value & G_TCON_TIMER_ENABLE) > (old_val &

                G_TCON_TIMER_ENABLE)) {

            exynos4210_gfrc_start(&s->g_timer);

        }

        if ((value & G_TCON_TIMER_ENABLE) < (old_val &

                G_TCON_TIMER_ENABLE)) {

            exynos4210_gfrc_stop(&s->g_timer);

        }



        /* Start CMP if transition from disabled to enabled */

        for (i = 0; i < MCT_GT_CMP_NUM; i++) {

            if ((value & G_TCON_COMP_ENABLE(i)) != (old_val &

                    G_TCON_COMP_ENABLE(i))) {

                exynos4210_gfrc_restart(s);

            }

        }

        break;



    case G_INT_CSTAT:

        s->g_timer.reg.int_cstat &= ~(value);

        for (i = 0; i < MCT_GT_CMP_NUM; i++) {

            if (value & G_INT_CSTAT_COMP(i)) {

                exynos4210_gcomp_lower_irq(&s->g_timer, i);

            }

        }

        break;



    case G_INT_ENB:



        /* Raise IRQ if transition from disabled to enabled and CSTAT pending */

        for (i = 0; i < MCT_GT_CMP_NUM; i++) {

            if ((value & G_INT_ENABLE(i)) > (s->g_timer.reg.tcon &

                    G_INT_ENABLE(i))) {

                if (s->g_timer.reg.int_cstat & G_INT_CSTAT_COMP(i)) {

                    exynos4210_gcomp_raise_irq(&s->g_timer, i);

                }

            }



            if ((value & G_INT_ENABLE(i)) < (s->g_timer.reg.tcon &

                    G_INT_ENABLE(i))) {

                exynos4210_gcomp_lower_irq(&s->g_timer, i);

            }

        }



        DPRINTF("global timer INT enable %llx\n", value);

        s->g_timer.reg.int_enb = value;

        break;



    case G_WSTAT:

        s->g_timer.reg.wstat &= ~(value);

        break;



    case G_COMP0_ADD_INCR: case G_COMP1_ADD_INCR:

    case G_COMP2_ADD_INCR: case G_COMP3_ADD_INCR:

        index = GET_G_COMP_ADD_INCR_IDX(offset);

        s->g_timer.reg.comp_add_incr[index] = value;

        s->g_timer.reg.wstat |= G_WSTAT_COMP_ADDINCR(index);

        break;



        /* Local timers */

    case L0_TCON: case L1_TCON:

        lt_i = GET_L_TIMER_IDX(offset);

        old_val = s->l_timer[lt_i].reg.tcon;



        s->l_timer[lt_i].reg.wstat |= L_WSTAT_TCON_WRITE;

        s->l_timer[lt_i].reg.tcon = value;



        /* Stop local CNT */

        if ((value & L_TCON_TICK_START) <

                (old_val & L_TCON_TICK_START)) {

            DPRINTF("local timer[%d] stop cnt\n", lt_i);

            exynos4210_ltick_cnt_stop(&s->l_timer[lt_i].tick_timer);

        }



        /* Stop local INT */

        if ((value & L_TCON_INT_START) <

                (old_val & L_TCON_INT_START)) {

            DPRINTF("local timer[%d] stop int\n", lt_i);

            exynos4210_ltick_int_stop(&s->l_timer[lt_i].tick_timer);

        }



        /* Start local CNT */

        if ((value & L_TCON_TICK_START) >

        (old_val & L_TCON_TICK_START)) {

            DPRINTF("local timer[%d] start cnt\n", lt_i);

            exynos4210_ltick_cnt_start(&s->l_timer[lt_i].tick_timer);

        }



        /* Start local INT */

        if ((value & L_TCON_INT_START) >

        (old_val & L_TCON_INT_START)) {

            DPRINTF("local timer[%d] start int\n", lt_i);

            exynos4210_ltick_int_start(&s->l_timer[lt_i].tick_timer);

        }



        /* Start or Stop local FRC if TCON changed */

        if ((value & L_TCON_FRC_START) >

        (s->l_timer[lt_i].reg.tcon & L_TCON_FRC_START)) {

            DPRINTF("local timer[%d] start frc\n", lt_i);

            exynos4210_lfrc_start(&s->l_timer[lt_i]);

        }

        if ((value & L_TCON_FRC_START) <

                (s->l_timer[lt_i].reg.tcon & L_TCON_FRC_START)) {

            DPRINTF("local timer[%d] stop frc\n", lt_i);

            exynos4210_lfrc_stop(&s->l_timer[lt_i]);

        }

        break;



    case L0_TCNTB: case L1_TCNTB:



        lt_i = GET_L_TIMER_IDX(offset);

        index = GET_L_TIMER_CNT_REG_IDX(offset, lt_i);



        /*

         * TCNTB is updated to internal register only after CNT expired.

         * Due to this we should reload timer to nearest moment when CNT is

         * expired and then in event handler update tcntb to new TCNTB value.

         */

        exynos4210_ltick_set_cntb(&s->l_timer[lt_i].tick_timer, value,

                s->l_timer[lt_i].tick_timer.icntb);



        s->l_timer[lt_i].reg.wstat |= L_WSTAT_TCNTB_WRITE;

        s->l_timer[lt_i].reg.cnt[L_REG_CNT_TCNTB] = value;



#ifdef DEBUG_MCT

        if (tcntb_min[lt_i] > value) {

            tcntb_min[lt_i] = value;

        }

        if (tcntb_max[lt_i] < value) {

            tcntb_max[lt_i] = value;

        }

        DPRINTF("local timer[%d] TCNTB write %llx; max=%x, min=%x\n",

                lt_i, value, tcntb_max[lt_i], tcntb_min[lt_i]);

#endif

        break;



    case L0_ICNTB: case L1_ICNTB:



        lt_i = GET_L_TIMER_IDX(offset);

        index = GET_L_TIMER_CNT_REG_IDX(offset, lt_i);



        s->l_timer[lt_i].reg.wstat |= L_WSTAT_ICNTB_WRITE;

        s->l_timer[lt_i].reg.cnt[L_REG_CNT_ICNTB] = value &

                ~L_ICNTB_MANUAL_UPDATE;



        /*

         * We need to avoid too small values for TCNTB*ICNTB. If not, IRQ event

         * could raise too fast disallowing QEMU to execute target code.

         */

        if (s->l_timer[lt_i].reg.cnt[L_REG_CNT_ICNTB] *

            s->l_timer[lt_i].reg.cnt[L_REG_CNT_TCNTB] < MCT_LT_CNT_LOW_LIMIT) {

            if (!s->l_timer[lt_i].reg.cnt[L_REG_CNT_TCNTB]) {

                s->l_timer[lt_i].reg.cnt[L_REG_CNT_ICNTB] =

                        MCT_LT_CNT_LOW_LIMIT;

            } else {

                s->l_timer[lt_i].reg.cnt[L_REG_CNT_ICNTB] =

                        MCT_LT_CNT_LOW_LIMIT /

                        s->l_timer[lt_i].reg.cnt[L_REG_CNT_TCNTB];

            }

        }



        if (value & L_ICNTB_MANUAL_UPDATE) {

            exynos4210_ltick_set_cntb(&s->l_timer[lt_i].tick_timer,

                    s->l_timer[lt_i].tick_timer.tcntb,

                    s->l_timer[lt_i].reg.cnt[L_REG_CNT_ICNTB]);

        }



#ifdef DEBUG_MCT

        if (icntb_min[lt_i] > value) {

            icntb_min[lt_i] = value;

        }

        if (icntb_max[lt_i] < value) {

            icntb_max[lt_i] = value;

        }

DPRINTF("local timer[%d] ICNTB write %llx; max=%x, min=%x\n\n",

        lt_i, value, icntb_max[lt_i], icntb_min[lt_i]);

#endif

break;



    case L0_FRCNTB: case L1_FRCNTB:



        lt_i = GET_L_TIMER_IDX(offset);

        index = GET_L_TIMER_CNT_REG_IDX(offset, lt_i);



        DPRINTF("local timer[%d] FRCNTB write %llx\n", lt_i, value);



        s->l_timer[lt_i].reg.wstat |= L_WSTAT_FRCCNTB_WRITE;

        s->l_timer[lt_i].reg.cnt[L_REG_CNT_FRCCNTB] = value;



        break;



    case L0_TCNTO: case L1_TCNTO:

    case L0_ICNTO: case L1_ICNTO:

    case L0_FRCNTO: case L1_FRCNTO:

        fprintf(stderr, "\n[exynos4210.mct: write to RO register "

                TARGET_FMT_plx "]\n\n", offset);

        break;



    case L0_INT_CSTAT: case L1_INT_CSTAT:

        lt_i = GET_L_TIMER_IDX(offset);



        DPRINTF("local timer[%d] CSTAT write %llx\n", lt_i, value);



        s->l_timer[lt_i].reg.int_cstat &= ~value;

        if (!s->l_timer[lt_i].reg.int_cstat) {

            qemu_irq_lower(s->l_timer[lt_i].irq);

        }

        break;



    case L0_INT_ENB: case L1_INT_ENB:

        lt_i = GET_L_TIMER_IDX(offset);

        old_val = s->l_timer[lt_i].reg.int_enb;



        /* Raise Local timer IRQ if cstat is pending */

        if ((value & L_INT_INTENB_ICNTEIE) > (old_val & L_INT_INTENB_ICNTEIE)) {

            if (s->l_timer[lt_i].reg.int_cstat & L_INT_CSTAT_INTCNT) {

                qemu_irq_raise(s->l_timer[lt_i].irq);

            }

        }



        s->l_timer[lt_i].reg.int_enb = value;



        break;



    case L0_WSTAT: case L1_WSTAT:

        lt_i = GET_L_TIMER_IDX(offset);



        s->l_timer[lt_i].reg.wstat &= ~value;

        break;



    default:

        hw_error("exynos4210.mct: bad write offset "

                TARGET_FMT_plx "\n", offset);

        break;

    }

}
