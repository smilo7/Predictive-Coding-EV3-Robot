#!/usr/bin/env python3
from models.kalman_filter_multisensor import run as run_kf2

from models.shared_functions import read_data_from_file_json, drive_motors

import json
import numpy as np
from ev3dev2.sensor import INPUT_1, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
#from ev3dev2.motor import OUTPUT_A, OUTPUT_C, MoveDifferential, SpeedRPM
#from ev3dev2.wheel import EV3EducationSetTire

############################
#        Load data         #
############################

"""
read in mean variance from the recordings
"""
data = read_data_from_file_json('data_out.json')

s1_variance = sum(data['s1']['variances'])/len(data['s1']['variances'])
s2_variance = sum(data['s2']['variances'])/len(data['s2']['variances'])
s3_variance = sum(data['s3']['variances'])/len(data['s3']['variances'])

print("variances s1:", s1_variance, "s2:", s2_variance, "s3:", s3_variance)


"""
Read in the lookup table for converting predicted light intensities
for the kalman filter
"""
lookup_tablejson = read_data_from_file_json('lookup_table.json')

# convert keys back into ints
lookup_table = {}
for k, v in lookup_tablejson.items():
    lookup_table[k] = {int(key): value for key, value in v.items()}

############################
#          SENSORS         #
############################
"""
initialise sensors
"""
l_sensor1 = ColorSensor(INPUT_1)
l_sensor2 = ColorSensor(INPUT_3)
us_sensor = UltrasonicSensor()




############################
#           RUN            #
############################

#hyperparams
N = 100

dist_intervals = np.arange(10, 110, 5)
dist_intervals = np.arange(10, 30, 5)
predictions = {'kf2':{}}

print("begin\n---------")
for dist in dist_intervals:
    x_kf2 = run_kf2(N, [s1_variance, s2_variance], [l_sensor1, l_sensor2], lookup_table)
    print(" prediction 2 sensor KF:", x_kf2[len(x_kf2)-1], "at", dist, "cm")
    print("\n")

    predictions['kf2'][str(dist)] = x_kf2

    drive_motors(-50)

print(predictions)

with open('kalman_preds.json', 'w') as outfile:
    json.dump(predictions, outfile)

print("done!")