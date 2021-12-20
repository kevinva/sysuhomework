void ich9_lpc_pm_init(PCIDevice *lpc_pci, bool smm_enabled)

{

    ICH9LPCState *lpc = ICH9_LPC_DEVICE(lpc_pci);

    qemu_irq sci_irq;



    sci_irq = qemu_allocate_irq(ich9_set_sci, lpc, 0);

    ich9_pm_init(lpc_pci, &lpc->pm, smm_enabled, sci_irq);

    ich9_lpc_reset(&lpc->d.qdev);

}
