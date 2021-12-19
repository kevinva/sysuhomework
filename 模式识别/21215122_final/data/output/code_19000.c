static int init_event_facility(SCLPEventFacility *event_facility)

{

    DeviceState *sdev = DEVICE(event_facility);

    DeviceState *quiesce;



    /* Spawn a new bus for SCLP events */

    qbus_create_inplace(&event_facility->sbus, sizeof(event_facility->sbus),

                        TYPE_SCLP_EVENTS_BUS, sdev, NULL);



    quiesce = qdev_create(&event_facility->sbus.qbus, "sclpquiesce");

    if (!quiesce) {

        return -1;

    }

    qdev_init_nofail(quiesce);



    object_initialize(&cpu_hotplug, sizeof(cpu_hotplug), TYPE_SCLP_CPU_HOTPLUG);

    qdev_set_parent_bus(DEVICE(&cpu_hotplug), BUS(&event_facility->sbus));

    object_property_set_bool(OBJECT(&cpu_hotplug), true, "realized", NULL);



    return 0;

}
