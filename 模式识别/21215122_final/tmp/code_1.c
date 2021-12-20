uint32_t HELPER(stfle)(CPUS390XState *env, uint64_t addr)

{

    uint64_t words[MAX_STFL_WORDS];

    unsigned count_m1 = env->regs[0] & 0xff;

    unsigned max_m1 = do_stfle(env, words);

    unsigned i;



    for (i = 0; i <= count_m1; ++i) {

        cpu_stq_data(env, addr + 8 * i, words[i]);

    }



    env->regs[0] = deposit64(env->regs[0], 0, 8, max_m1);

    return (count_m1 >= max_m1 ? 0 : 3);

}
