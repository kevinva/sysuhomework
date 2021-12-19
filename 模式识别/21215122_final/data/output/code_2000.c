static void handle_port_status_write(EHCIState *s, int port, uint32_t val)

{

    uint32_t *portsc = &s->portsc[port];

    USBDevice *dev = s->ports[port].dev;



    /* Clear rwc bits */

    *portsc &= ~(val & PORTSC_RWC_MASK);

    /* The guest may clear, but not set the PED bit */

    *portsc &= val | ~PORTSC_PED;

    /* POWNER is masked out by RO_MASK as it is RO when we've no companion */

    handle_port_owner_write(s, port, val);

    /* And finally apply RO_MASK */

    val &= PORTSC_RO_MASK;



    if ((val & PORTSC_PRESET) && !(*portsc & PORTSC_PRESET)) {

        trace_usb_ehci_port_reset(port, 1);

    }



    if (!(val & PORTSC_PRESET) &&(*portsc & PORTSC_PRESET)) {

        trace_usb_ehci_port_reset(port, 0);

        if (dev && dev->attached) {

            usb_port_reset(&s->ports[port]);

            *portsc &= ~PORTSC_CSC;

        }



        /*

         *  Table 2.16 Set the enable bit(and enable bit change) to indicate

         *  to SW that this port has a high speed device attached

         */

        if (dev && dev->attached && (dev->speedmask & USB_SPEED_MASK_HIGH)) {

            val |= PORTSC_PED;

        }

    }



    *portsc &= ~PORTSC_RO_MASK;

    *portsc |= val;

}
