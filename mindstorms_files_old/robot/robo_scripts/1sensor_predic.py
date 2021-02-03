#!/usr/bin/env python3
import numpy as np
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
from ev3dev2.button import Button
from ev3dev2.display import Display
import time

"""load up peripherals. sensors motors buttons etc"""
button = Button()
l_sensor = ColorSensor()
u_sensor = UltrasonicSensor()

def return_pressed_button(button):
    """waits for center button to be pressed and then returns the button object"""
    
    pressed_button = None
    while True:
        if button.enter:
            print(button.enter)
            pressed_button = button
            break
    return True, pressed_button
        

def record_light(duration=3):
    """records light levels for given number of secs seconds and calculates average"""
    start_time = time.time()
    t_elapsed = time.time() - start_time
    
    l_arr = []
    while t_elapsed < duration:
        l_arr.append(l_sensor.ambient_light_intensity)
        t_elapsed = time.time() - start_time

    return l_arr


def ave_l_levels():
    """gets the mean light reading. ideally robot will be stationary so it is just for that distance"""
    #start program
    pressed, button = return_pressed_button(button)
    mean = 0
    if pressed:
        l_levels = record_light()
        mean = sum(l_levels)/len(l_levels)
    return mean
