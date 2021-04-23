#!/usr/bin/env python3
import numpy as np
import pandas as pd
import json
from ev3dev2.sensor import INPUT_1, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
from ev3dev2.button import Button
from ev3dev2.sound import Sound
from ev3dev2.motor import OUTPUT_A, OUTPUT_C, MoveDifferential, SpeedRPM
from ev3dev2.wheel import EV3EducationSetTire
import time


"""load up peripherals. sensors motors buttons etc"""
button = Button()
l_sensor2 = ColorSensor(INPUT_3)
l_sensor1 = ColorSensor(INPUT_1)
us_sensor = UltrasonicSensor()
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
        

def record_light(N=1000):
    """
    records light levels for given number of seconds and calculates average
    N - number of recordings to take
    returns l_readings, dict of means
    """
    start_time = time.time()
    t_elapsed = time.time() - start_time

    l_readings = [[],[],[]]
    #l_readings = {'s1':{'means':[ ], 'variances':[]}, 's2':{'means':[], 'variances':[]}}
    
    for i in range(0, N):
        l_readings[0].append(l_sensor1.ambient_light_intensity)
        l_readings[1].append(l_sensor2.ambient_light_intensity)
        l_readings[2].append(us_sensor.distance_centimeters)

    """
    for key, val in l_readings.items():
        if key == 's1':
            val['means'] = l_sensor1.ambient_light_intensity
        elif key == 's2':
            val['means'] = l_sensor3.ambient_light_intensity
    """

    t_elapsed = time.time() - start_time
    print(N, "recordings done in ", t_elapsed)
    return l_readings


def get_l_levels():
    """gets the light levels if the button has been pressed"""
    #start program
    pressed, btn = return_pressed_button(button)
    l_levels = 0
    if pressed:
        l_levels = record_light()
    return l_levels

def drive_motors(distance):
    movediff = MoveDifferential(OUTPUT_A, OUTPUT_C, EV3EducationSetTire, 10 * 8)
    movediff.on_for_distance(SpeedRPM(20), distance)
    
def calc_mean(data):
    """
    calculate means for light recordings
    data format {'s1':{'means':[], 'variances':[]}, 's2':{'means':[], 'variances':[]}}
    """
    return sum(data)/len(data)

def calc_variance(data):
    """
    Calculate the variance for a given array of po = s1_means[1:]
distances = distances[1:]ints
    input: data (array)
    return: variance
    """
    mean = sum(data) / len(data)
    deviations = [(x - mean) ** 2 for x in data]
    variance = sum(deviations) / len(data)
    return variance

def write_results_to_csv(data):
    """
    Dumps to results to a csv file so that the robot can use it when running pp_unit program
    """
    writeme = np.asarray(data)
    writeme.tofile('data.csv', sep=',', format='%10.5f')
    print('Done, written results to file :)')

def write_dict_to_json(dic_data, filename):
    """
    Using json write the dictionary to a json file
    """
    with open(filename, 'w') as outfile:
        json.dump(dic_data, outfile)

 
"""
Calculate mean and variance light intensity over several distances
Robot should be moved to the correct distance by hand
"""


NUM_INTERVALS = (20) #numebr of distance intervals 150cm / 20

data_out = {'s1':{'means':[], 'variances':[]}, 's2':{'means':[], 'variances':[]}, 's3':{'means':[], 'variances':[]}}


dist_intervals = np.arange(10, 115, 5)
#dist_intervals = np.arange(20, 85, 5)

#for i in range(0, NUM_INTERVALS):
for i, dist in enumerate(dist_intervals):
    #l_levels = get_l_levels()
    print(i, "out of", len(dist_intervals))
    l_levels = record_light(1000)
    #print(l_levels)
    data_out['s1']['means'].append(calc_mean(l_levels[0]))
    data_out['s2']['means'].append(calc_mean(l_levels[1]))
    data_out['s3']['means'].append(calc_mean(l_levels[2]))

    data_out['s1']['variances'].append(calc_variance(l_levels[0]))
    data_out['s2']['variances'].append(calc_variance(l_levels[1]))
    data_out['s3']['variances'].append(calc_variance(l_levels[2]))
    #then drive motor for specifed distance
    drive_motors(-50)
    time.sleep(0.1)

#back to begining
#time.sleep(1)
#drive_motors(500)
#drive_motors(-50*-NUM_INTERVALS)

print(data_out)
write_dict_to_json(data_out, 'data_out.json')
#print("results", means, variances)
#write_results_to_csv(means)
#write_results_to_csv([means, variances])