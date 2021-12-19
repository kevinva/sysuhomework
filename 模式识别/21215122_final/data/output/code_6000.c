QEMUClock *qemu_clock_ptr(QEMUClockType type)

{

    return &qemu_clocks[type];

}
