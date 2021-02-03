#!/usr/bin/env python3

from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
import ev3dev2.fonts as fonts
from ev3dev2.display import Display
import time

display = Display()
s_color = ColorSensor() #color sensor we will be using for light sensitivity
s_us = UltrasonicSensor()


while True:
    distance = s_us.distance_centimeters    
    reading  = s_color.ambient_light_intensity

    print("Dist", distance, "Light", reading)
    #display.draw.text((10,10), str(reading), font=fonts.load('luBS14'))