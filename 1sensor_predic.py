#!/usr/bin/env python3
import numpy as np
from ev3dev2.sensor import INPUT_1, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
from ev3dev2.button import Button
from ev3dev2.sound import Sound
import time


"""load up peripherals. sensors motors buttons etc"""
button = Button()
l_sensor3 = ColorSensor(INPUT_3)
l_sensor1 = ColorSensor(INPUT_1)
u_sensor = UltrasonicSensor()
sound = Sound()

def return_pressed_button(button):
    """waits for center button to be pressed and then returns the button object"""
    
    pressed_button = None
    while True:
        sound.play_tone(400, 0.5)
        #print(button.buttons_pressed)
        print("press button:")
        if button.enter:
            print(button.enter)
            sound.play_tone(300, 0.5)
            pressed_button = button
            break
    return True, pressed_button
        

def record_light(duration=3, N=1000):
    """records light levels for given number of secs seconds and calculates average"""
    start_time = time.time()
    t_elapsed = time.time() - start_time
    #l_sensor = ColorSensor(INPUT_1)
    l_arr = []
    for i in range(0, N):
    #while t_elapsed < duration:
        l_arr.append(l_sensor1.ambient_light_intensity)
        t_elapsed = time.time() - start_time
    print(N, "recordings done in ", t_elapsed)
    return l_arr


def get_l_levels():
    """gets the light levels if the button has been pressed"""
    #start program
    pressed, btn = return_pressed_button(button)
    l_levels = 0
    if pressed:
        l_levels = record_light()
    return l_levels

def calc_mean(data):
    return sum(data)/len(data)

def calc_variance(data):
    """
    Calculate the variance for a given array of points
    input: data (array)
    return: variance
    """
    mean = sum(data) / len(data)
    deviations = [(x-mean) ** 2 for x in data]
    variance = sum(deviations) /len(data)
    return variance

def write_results_to_csv(data):
    """
    Dumps to results to a csv file so that the robot can use it when running pp_unit program
    """
    writeme = np.asarray(data)
    writeme.tofile('data.csv', sep=',', format='%10.5f')

 
"""
Calculate mean and variance light intensity over several distances
Robot should be moved to the correct distance by hand
"""


NUM_INTERVALS = (20) #numebr of distance intervals 150cm / 20
means = []
variances = []
for i in range(0, NUM_INTERVALS):
    l_levels = get_l_levels()
    means.append(calc_mean(l_levels))
    variances.append(calc_variance(l_levels))

print("results", means, variances)
write_results_to_csv(means)
#write_results_to_csv([means, variances])



"""
In this example we infer one hidden state distance X
Robot is to remain static in one location
We assume the generative model has already been learned (ie how light reading is related to distance)
"""