#!/usr/bin/env python3


from ev3dev2.motor import OUTPUT_A, OUTPUT_C, MoveDifferential, SpeedRPM
from ev3dev2.wheel import EV3EducationSetTire

import time

def drive_motors(distance):
    movediff = MoveDifferential(OUTPUT_A, OUTPUT_C, EV3EducationSetTire, 10 * 8)
    movediff.on_for_distance(SpeedRPM(20), distance)


for i in range(0, 10):
    drive_motors(-50)
    time.sleep(5)
