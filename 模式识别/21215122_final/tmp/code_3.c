static void disas_thumb_insn(CPUARMState *env, DisasContext *s)

{

    uint32_t val, insn, op, rm, rn, rd, shift, cond;

    int32_t offset;

    int i;

    TCGv_i32 tmp;

    TCGv_i32 tmp2;

    TCGv_i32 addr;



    if (s->condexec_mask) {

        cond = s->condexec_cond;

        if (cond != 0x0e) {     /* Skip conditional when condition is AL. */

          s->condlabel = gen_new_label();

          gen_test_cc(cond ^ 1, s->condlabel);

          s->condjmp = 1;

        }

    }



    insn = arm_lduw_code(env, s->pc, s->bswap_code);

    s->pc += 2;



    switch (insn >> 12) {

    case 0: case 1:



        rd = insn & 7;

        op = (insn >> 11) & 3;

        if (op == 3) {

            /* add/subtract */

            rn = (insn >> 3) & 7;

            tmp = load_reg(s, rn);

            if (insn & (1 << 10)) {

                /* immediate */

                tmp2 = tcg_temp_new_i32();

                tcg_gen_movi_i32(tmp2, (insn >> 6) & 7);

            } else {

                /* reg */

                rm = (insn >> 6) & 7;

                tmp2 = load_reg(s, rm);

            }

            if (insn & (1 << 9)) {

                if (s->condexec_mask)

                    tcg_gen_sub_i32(tmp, tmp, tmp2);

                else

                    gen_sub_CC(tmp, tmp, tmp2);

            } else {

                if (s->condexec_mask)

                    tcg_gen_add_i32(tmp, tmp, tmp2);

                else

                    gen_add_CC(tmp, tmp, tmp2);

            }

            tcg_temp_free_i32(tmp2);

            store_reg(s, rd, tmp);

        } else {

            /* shift immediate */

            rm = (insn >> 3) & 7;

            shift = (insn >> 6) & 0x1f;

            tmp = load_reg(s, rm);

            gen_arm_shift_im(tmp, op, shift, s->condexec_mask == 0);

            if (!s->condexec_mask)

                gen_logic_CC(tmp);

            store_reg(s, rd, tmp);

        }

        break;

    case 2: case 3:

        /* arithmetic large immediate */

        op = (insn >> 11) & 3;

        rd = (insn >> 8) & 0x7;

        if (op == 0) { /* mov */

            tmp = tcg_temp_new_i32();

            tcg_gen_movi_i32(tmp, insn & 0xff);

            if (!s->condexec_mask)

                gen_logic_CC(tmp);

            store_reg(s, rd, tmp);

        } else {

            tmp = load_reg(s, rd);

            tmp2 = tcg_temp_new_i32();

            tcg_gen_movi_i32(tmp2, insn & 0xff);

            switch (op) {

            case 1: /* cmp */

                gen_sub_CC(tmp, tmp, tmp2);

                tcg_temp_free_i32(tmp);

                tcg_temp_free_i32(tmp2);

                break;

            case 2: /* add */

                if (s->condexec_mask)

                    tcg_gen_add_i32(tmp, tmp, tmp2);

                else

                    gen_add_CC(tmp, tmp, tmp2);

                tcg_temp_free_i32(tmp2);

                store_reg(s, rd, tmp);

                break;

            case 3: /* sub */

                if (s->condexec_mask)

                    tcg_gen_sub_i32(tmp, tmp, tmp2);

                else

                    gen_sub_CC(tmp, tmp, tmp2);

                tcg_temp_free_i32(tmp2);

                store_reg(s, rd, tmp);

                break;

            }

        }

        break;

    case 4:

        if (insn & (1 << 11)) {

            rd = (insn >> 8) & 7;

            /* load pc-relative.  Bit 1 of PC is ignored.  */

            val = s->pc + 2 + ((insn & 0xff) * 4);

            val &= ~(uint32_t)2;

            addr = tcg_temp_new_i32();

            tcg_gen_movi_i32(addr, val);

            tmp = tcg_temp_new_i32();

            gen_aa32_ld32u(tmp, addr, IS_USER(s));

            tcg_temp_free_i32(addr);

            store_reg(s, rd, tmp);

            break;

        }

        if (insn & (1 << 10)) {

            /* data processing extended or blx */

            rd = (insn & 7) | ((insn >> 4) & 8);

            rm = (insn >> 3) & 0xf;

            op = (insn >> 8) & 3;

            switch (op) {

            case 0: /* add */

                tmp = load_reg(s, rd);

                tmp2 = load_reg(s, rm);

                tcg_gen_add_i32(tmp, tmp, tmp2);

                tcg_temp_free_i32(tmp2);

                store_reg(s, rd, tmp);

                break;

            case 1: /* cmp */

                tmp = load_reg(s, rd);

                tmp2 = load_reg(s, rm);

                gen_sub_CC(tmp, tmp, tmp2);

                tcg_temp_free_i32(tmp2);

                tcg_temp_free_i32(tmp);

                break;

            case 2: /* mov/cpy */

                tmp = load_reg(s, rm);

                store_reg(s, rd, tmp);

                break;

            case 3:/* branch [and link] exchange thumb register */

                tmp = load_reg(s, rm);

                if (insn & (1 << 7)) {

                    ARCH(5);

                    val = (uint32_t)s->pc | 1;

                    tmp2 = tcg_temp_new_i32();

                    tcg_gen_movi_i32(tmp2, val);

                    store_reg(s, 14, tmp2);

                }

                /* already thumb, no need to check */

                gen_bx(s, tmp);

                break;

            }

            break;

        }



        /* data processing register */

        rd = insn & 7;

        rm = (insn >> 3) & 7;

        op = (insn >> 6) & 0xf;

        if (op == 2 || op == 3 || op == 4 || op == 7) {

            /* the shift/rotate ops want the operands backwards */

            val = rm;

            rm = rd;

            rd = val;

            val = 1;

        } else {

            val = 0;

        }



        if (op == 9) { /* neg */

            tmp = tcg_temp_new_i32();

            tcg_gen_movi_i32(tmp, 0);

        } else if (op != 0xf) { /* mvn doesn't read its first operand */

            tmp = load_reg(s, rd);

        } else {

            TCGV_UNUSED_I32(tmp);

        }



        tmp2 = load_reg(s, rm);

        switch (op) {

        case 0x0: /* and */

            tcg_gen_and_i32(tmp, tmp, tmp2);

            if (!s->condexec_mask)

                gen_logic_CC(tmp);

            break;

        case 0x1: /* eor */

            tcg_gen_xor_i32(tmp, tmp, tmp2);

            if (!s->condexec_mask)

                gen_logic_CC(tmp);

            break;

        case 0x2: /* lsl */

            if (s->condexec_mask) {

                gen_shl(tmp2, tmp2, tmp);

            } else {

                gen_helper_shl_cc(tmp2, cpu_env, tmp2, tmp);

                gen_logic_CC(tmp2);

            }

            break;

        case 0x3: /* lsr */

            if (s->condexec_mask) {

                gen_shr(tmp2, tmp2, tmp);

            } else {

                gen_helper_shr_cc(tmp2, cpu_env, tmp2, tmp);

                gen_logic_CC(tmp2);

            }

            break;

        case 0x4: /* asr */

            if (s->condexec_mask) {

                gen_sar(tmp2, tmp2, tmp);

            } else {

                gen_helper_sar_cc(tmp2, cpu_env, tmp2, tmp);

                gen_logic_CC(tmp2);

            }

            break;

        case 0x5: /* adc */

            if (s->condexec_mask) {

                gen_adc(tmp, tmp2);

            } else {

                gen_adc_CC(tmp, tmp, tmp2);

            }

            break;

        case 0x6: /* sbc */

            if (s->condexec_mask) {

                gen_sub_carry(tmp, tmp, tmp2);

            } else {

                gen_sbc_CC(tmp, tmp, tmp2);

            }

            break;

        case 0x7: /* ror */

            if (s->condexec_mask) {

                tcg_gen_andi_i32(tmp, tmp, 0x1f);

                tcg_gen_rotr_i32(tmp2, tmp2, tmp);

            } else {

                gen_helper_ror_cc(tmp2, cpu_env, tmp2, tmp);

                gen_logic_CC(tmp2);

            }

            break;

        case 0x8: /* tst */

            tcg_gen_and_i32(tmp, tmp, tmp2);

            gen_logic_CC(tmp);

            rd = 16;

            break;

        case 0x9: /* neg */

            if (s->condexec_mask)

                tcg_gen_neg_i32(tmp, tmp2);

            else

                gen_sub_CC(tmp, tmp, tmp2);

            break;

        case 0xa: /* cmp */

            gen_sub_CC(tmp, tmp, tmp2);

            rd = 16;

            break;

        case 0xb: /* cmn */

            gen_add_CC(tmp, tmp, tmp2);

            rd = 16;

            break;

        case 0xc: /* orr */

            tcg_gen_or_i32(tmp, tmp, tmp2);

            if (!s->condexec_mask)

                gen_logic_CC(tmp);

            break;

        case 0xd: /* mul */

            tcg_gen_mul_i32(tmp, tmp, tmp2);

            if (!s->condexec_mask)

                gen_logic_CC(tmp);

            break;

        case 0xe: /* bic */

            tcg_gen_andc_i32(tmp, tmp, tmp2);

            if (!s->condexec_mask)

                gen_logic_CC(tmp);

            break;

        case 0xf: /* mvn */

            tcg_gen_not_i32(tmp2, tmp2);

            if (!s->condexec_mask)

                gen_logic_CC(tmp2);

            val = 1;

            rm = rd;

            break;

        }

        if (rd != 16) {

            if (val) {

                store_reg(s, rm, tmp2);

                if (op != 0xf)

                    tcg_temp_free_i32(tmp);

            } else {

                store_reg(s, rd, tmp);

                tcg_temp_free_i32(tmp2);

            }

        } else {

            tcg_temp_free_i32(tmp);

            tcg_temp_free_i32(tmp2);

        }

        break;



    case 5:

        /* load/store register offset.  */

        rd = insn & 7;

        rn = (insn >> 3) & 7;

        rm = (insn >> 6) & 7;

        op = (insn >> 9) & 7;

        addr = load_reg(s, rn);

        tmp = load_reg(s, rm);

        tcg_gen_add_i32(addr, addr, tmp);

        tcg_temp_free_i32(tmp);



        if (op < 3) { /* store */

            tmp = load_reg(s, rd);

        } else {

            tmp = tcg_temp_new_i32();

        }



        switch (op) {

        case 0: /* str */

            gen_aa32_st32(tmp, addr, IS_USER(s));

            break;

        case 1: /* strh */

            gen_aa32_st16(tmp, addr, IS_USER(s));

            break;

        case 2: /* strb */

            gen_aa32_st8(tmp, addr, IS_USER(s));

            break;

        case 3: /* ldrsb */

            gen_aa32_ld8s(tmp, addr, IS_USER(s));

            break;

        case 4: /* ldr */

            gen_aa32_ld32u(tmp, addr, IS_USER(s));

            break;

        case 5: /* ldrh */

            gen_aa32_ld16u(tmp, addr, IS_USER(s));

            break;

        case 6: /* ldrb */

            gen_aa32_ld8u(tmp, addr, IS_USER(s));

            break;

        case 7: /* ldrsh */

            gen_aa32_ld16s(tmp, addr, IS_USER(s));

            break;

        }

        if (op >= 3) { /* load */

            store_reg(s, rd, tmp);

        } else {

            tcg_temp_free_i32(tmp);

        }

        tcg_temp_free_i32(addr);

        break;



    case 6:

        /* load/store word immediate offset */

        rd = insn & 7;

        rn = (insn >> 3) & 7;

        addr = load_reg(s, rn);

        val = (insn >> 4) & 0x7c;

        tcg_gen_addi_i32(addr, addr, val);



        if (insn & (1 << 11)) {

            /* load */

            tmp = tcg_temp_new_i32();

            gen_aa32_ld32u(tmp, addr, IS_USER(s));

            store_reg(s, rd, tmp);

        } else {

            /* store */

            tmp = load_reg(s, rd);

            gen_aa32_st32(tmp, addr, IS_USER(s));

            tcg_temp_free_i32(tmp);

        }

        tcg_temp_free_i32(addr);

        break;



    case 7:

        /* load/store byte immediate offset */

        rd = insn & 7;

        rn = (insn >> 3) & 7;

        addr = load_reg(s, rn);

        val = (insn >> 6) & 0x1f;

        tcg_gen_addi_i32(addr, addr, val);



        if (insn & (1 << 11)) {

            /* load */

            tmp = tcg_temp_new_i32();

            gen_aa32_ld8u(tmp, addr, IS_USER(s));

            store_reg(s, rd, tmp);

        } else {

            /* store */

            tmp = load_reg(s, rd);

            gen_aa32_st8(tmp, addr, IS_USER(s));

            tcg_temp_free_i32(tmp);

        }

        tcg_temp_free_i32(addr);

        break;



    case 8:

        /* load/store halfword immediate offset */

        rd = insn & 7;

        rn = (insn >> 3) & 7;

        addr = load_reg(s, rn);

        val = (insn >> 5) & 0x3e;

        tcg_gen_addi_i32(addr, addr, val);



        if (insn & (1 << 11)) {

            /* load */

            tmp = tcg_temp_new_i32();

            gen_aa32_ld16u(tmp, addr, IS_USER(s));

            store_reg(s, rd, tmp);

        } else {

            /* store */

            tmp = load_reg(s, rd);

            gen_aa32_st16(tmp, addr, IS_USER(s));

            tcg_temp_free_i32(tmp);

        }

        tcg_temp_free_i32(addr);

        break;



    case 9:

        /* load/store from stack */

        rd = (insn >> 8) & 7;

        addr = load_reg(s, 13);

        val = (insn & 0xff) * 4;

        tcg_gen_addi_i32(addr, addr, val);



        if (insn & (1 << 11)) {

            /* load */

            tmp = tcg_temp_new_i32();

            gen_aa32_ld32u(tmp, addr, IS_USER(s));

            store_reg(s, rd, tmp);

        } else {

            /* store */

            tmp = load_reg(s, rd);

            gen_aa32_st32(tmp, addr, IS_USER(s));

            tcg_temp_free_i32(tmp);

        }

        tcg_temp_free_i32(addr);

        break;



    case 10:

        /* add to high reg */

        rd = (insn >> 8) & 7;

        if (insn & (1 << 11)) {

            /* SP */

            tmp = load_reg(s, 13);

        } else {

            /* PC. bit 1 is ignored.  */

            tmp = tcg_temp_new_i32();

            tcg_gen_movi_i32(tmp, (s->pc + 2) & ~(uint32_t)2);

        }

        val = (insn & 0xff) * 4;

        tcg_gen_addi_i32(tmp, tmp, val);

        store_reg(s, rd, tmp);

        break;



    case 11:

        /* misc */

        op = (insn >> 8) & 0xf;

        switch (op) {

        case 0:

            /* adjust stack pointer */

            tmp = load_reg(s, 13);

            val = (insn & 0x7f) * 4;

            if (insn & (1 << 7))

                val = -(int32_t)val;

            tcg_gen_addi_i32(tmp, tmp, val);

            store_reg(s, 13, tmp);

            break;



        case 2: /* sign/zero extend.  */

            ARCH(6);

            rd = insn & 7;

            rm = (insn >> 3) & 7;

            tmp = load_reg(s, rm);

            switch ((insn >> 6) & 3) {

            case 0: gen_sxth(tmp); break;

            case 1: gen_sxtb(tmp); break;

            case 2: gen_uxth(tmp); break;

            case 3: gen_uxtb(tmp); break;

            }

            store_reg(s, rd, tmp);

            break;

        case 4: case 5: case 0xc: case 0xd:

            /* push/pop */

            addr = load_reg(s, 13);

            if (insn & (1 << 8))

                offset = 4;

            else

                offset = 0;

            for (i = 0; i < 8; i++) {

                if (insn & (1 << i))

                    offset += 4;

            }

            if ((insn & (1 << 11)) == 0) {

                tcg_gen_addi_i32(addr, addr, -offset);

            }

            for (i = 0; i < 8; i++) {

                if (insn & (1 << i)) {

                    if (insn & (1 << 11)) {

                        /* pop */

                        tmp = tcg_temp_new_i32();

                        gen_aa32_ld32u(tmp, addr, IS_USER(s));

                        store_reg(s, i, tmp);

                    } else {

                        /* push */

                        tmp = load_reg(s, i);

                        gen_aa32_st32(tmp, addr, IS_USER(s));

                        tcg_temp_free_i32(tmp);

                    }

                    /* advance to the next address.  */

                    tcg_gen_addi_i32(addr, addr, 4);

                }

            }

            TCGV_UNUSED_I32(tmp);

            if (insn & (1 << 8)) {

                if (insn & (1 << 11)) {

                    /* pop pc */

                    tmp = tcg_temp_new_i32();

                    gen_aa32_ld32u(tmp, addr, IS_USER(s));

                    /* don't set the pc until the rest of the instruction

                       has completed */

                } else {

                    /* push lr */

                    tmp = load_reg(s, 14);

                    gen_aa32_st32(tmp, addr, IS_USER(s));

                    tcg_temp_free_i32(tmp);

                }

                tcg_gen_addi_i32(addr, addr, 4);

            }

            if ((insn & (1 << 11)) == 0) {

                tcg_gen_addi_i32(addr, addr, -offset);

            }

            /* write back the new stack pointer */

            store_reg(s, 13, addr);

            /* set the new PC value */

            if ((insn & 0x0900) == 0x0900) {

                store_reg_from_load(env, s, 15, tmp);

            }

            break;



        case 1: case 3: case 9: case 11: /* czb */

            rm = insn & 7;

            tmp = load_reg(s, rm);

            s->condlabel = gen_new_label();

            s->condjmp = 1;

            if (insn & (1 << 11))

                tcg_gen_brcondi_i32(TCG_COND_EQ, tmp, 0, s->condlabel);

            else

                tcg_gen_brcondi_i32(TCG_COND_NE, tmp, 0, s->condlabel);

            tcg_temp_free_i32(tmp);

            offset = ((insn & 0xf8) >> 2) | (insn & 0x200) >> 3;

            val = (uint32_t)s->pc + 2;

            val += offset;

            gen_jmp(s, val);

            break;



        case 15: /* IT, nop-hint.  */

            if ((insn & 0xf) == 0) {

                gen_nop_hint(s, (insn >> 4) & 0xf);

                break;

            }

            /* If Then.  */

            s->condexec_cond = (insn >> 4) & 0xe;

            s->condexec_mask = insn & 0x1f;

            /* No actual code generated for this insn, just setup state.  */

            break;



        case 0xe: /* bkpt */

            ARCH(5);

            gen_exception_insn(s, 2, EXCP_BKPT);

            break;



        case 0xa: /* rev */

            ARCH(6);

            rn = (insn >> 3) & 0x7;

            rd = insn & 0x7;

            tmp = load_reg(s, rn);

            switch ((insn >> 6) & 3) {

            case 0: tcg_gen_bswap32_i32(tmp, tmp); break;

            case 1: gen_rev16(tmp); break;

            case 3: gen_revsh(tmp); break;

            default: goto illegal_op;

            }

            store_reg(s, rd, tmp);

            break;



        case 6:

            switch ((insn >> 5) & 7) {

            case 2:

                /* setend */

                ARCH(6);

                if (((insn >> 3) & 1) != s->bswap_code) {

                    /* Dynamic endianness switching not implemented. */

                    qemu_log_mask(LOG_UNIMP, "arm: unimplemented setend\n");

                    goto illegal_op;

                }

                break;

            case 3:

                /* cps */

                ARCH(6);

                if (IS_USER(s)) {

                    break;

                }

                if (IS_M(env)) {

                    tmp = tcg_const_i32((insn & (1 << 4)) != 0);

                    /* FAULTMASK */

                    if (insn & 1) {

                        addr = tcg_const_i32(19);

                        gen_helper_v7m_msr(cpu_env, addr, tmp);

                        tcg_temp_free_i32(addr);

                    }

                    /* PRIMASK */

                    if (insn & 2) {

                        addr = tcg_const_i32(16);

                        gen_helper_v7m_msr(cpu_env, addr, tmp);

                        tcg_temp_free_i32(addr);

                    }

                    tcg_temp_free_i32(tmp);

                    gen_lookup_tb(s);

                } else {

                    if (insn & (1 << 4)) {

                        shift = CPSR_A | CPSR_I | CPSR_F;

                    } else {

                        shift = 0;

                    }

                    gen_set_psr_im(s, ((insn & 7) << 6), 0, shift);

                }

                break;

            default:

                goto undef;

            }

            break;



        default:

            goto undef;

        }

        break;



    case 12:

    {

        /* load/store multiple */

        TCGv_i32 loaded_var;

        TCGV_UNUSED_I32(loaded_var);

        rn = (insn >> 8) & 0x7;

        addr = load_reg(s, rn);

        for (i = 0; i < 8; i++) {

            if (insn & (1 << i)) {

                if (insn & (1 << 11)) {

                    /* load */

                    tmp = tcg_temp_new_i32();

                    gen_aa32_ld32u(tmp, addr, IS_USER(s));

                    if (i == rn) {

                        loaded_var = tmp;

                    } else {

                        store_reg(s, i, tmp);

                    }

                } else {

                    /* store */

                    tmp = load_reg(s, i);

                    gen_aa32_st32(tmp, addr, IS_USER(s));

                    tcg_temp_free_i32(tmp);

                }

                /* advance to the next address */

                tcg_gen_addi_i32(addr, addr, 4);

            }

        }

        if ((insn & (1 << rn)) == 0) {

            /* base reg not in list: base register writeback */

            store_reg(s, rn, addr);

        } else {

            /* base reg in list: if load, complete it now */

            if (insn & (1 << 11)) {

                store_reg(s, rn, loaded_var);

            }

            tcg_temp_free_i32(addr);

        }

        break;

    }

    case 13:

        /* conditional branch or swi */

        cond = (insn >> 8) & 0xf;

        if (cond == 0xe)

            goto undef;



        if (cond == 0xf) {

            /* swi */

            gen_set_pc_im(s, s->pc);

            s->is_jmp = DISAS_SWI;

            break;

        }

        /* generate a conditional jump to next instruction */

        s->condlabel = gen_new_label();

        gen_test_cc(cond ^ 1, s->condlabel);

        s->condjmp = 1;



        /* jump to the offset */

        val = (uint32_t)s->pc + 2;

        offset = ((int32_t)insn << 24) >> 24;

        val += offset << 1;

        gen_jmp(s, val);

        break;



    case 14:

        if (insn & (1 << 11)) {

            if (disas_thumb2_insn(env, s, insn))

              goto undef32;

            break;

        }

        /* unconditional branch */

        val = (uint32_t)s->pc;

        offset = ((int32_t)insn << 21) >> 21;

        val += (offset << 1) + 2;

        gen_jmp(s, val);

        break;



    case 15:

        if (disas_thumb2_insn(env, s, insn))

            goto undef32;

        break;

    }

    return;

undef32:

    gen_exception_insn(s, 4, EXCP_UDEF);

    return;

illegal_op:

undef:

    gen_exception_insn(s, 2, EXCP_UDEF);

}
