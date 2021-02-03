#!/usr/bin/env python3

from ev3dev2.motor import  LargeMotor, OUTPUT_A, OUTPUT_B, SpeedPercent, MoveTank
from ev3dev2.sensor.lego import UltrasonicSensor
from ev3dev2.led import Leds
import time

from ev3dev2.sound import Sound

sound = Sound()
sound.speak('Watch out mother fuckers I am coming to fuck your momma')

tank_drive = MoveTank(OUTPUT_A, OUTPUT_B)
us = UltrasonicSensor()
leds = Leds()


wall = False

while wall != True:
    distance = us.distance_centimeters
    if distance < 10:
        wall = True
    #otherwise continue driving    
    tank_drive.on(50,50)

tank_drive.stop()


