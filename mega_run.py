#!/usr/bin/env python3
from models.one_sensor import run as run1
from models.two_sensor import run as run2
from models.three_sensor import run as run3

from models.shared_functions import read_data_from_file_json, drive_motors

import json
import numpy as np
from ev3dev2.sensor import INPUT_1, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
#from ev3dev2.motor import OUTPUT_A, OUTPUT_C, MoveDifferential, SpeedRPM
#from ev3dev2.wheel import EV3EducationSetTire


"""
read in mean variance from the recordings
"""
data = read_data_from_file_json('data_out.json')

s1_variance = sum(data['s1']['variances'])/len(data['s1']['variances'])
s2_variance = sum(data['s2']['variances'])/len(data['s2']['variances'])
s3_variance = sum(data['s3']['variances'])/len(data['s3']['variances'])

print("variances s1:", s1_variance, "s2:", s2_variance, "s3:", s3_variance)

"""
read generative mapping params from file
"""
params = read_data_from_file_json('genmap_params.json')

#assign params to globals
s1params_a, s1params_b, s1params_c = params['s1']['a'], params['s1']['b'], params['s1']['c']
s2params_a, s2params_b, s2params_c = params['s2']['a'], params['s2']['b'], params['s2']['c']
s3params_a, s3params_b = params['s3']['a'], params['s3']['b']

g_params = [
            s1params_a, s1params_b, s1params_c,
            s2params_a, s2params_b, s2params_c,
            s3params_a, s3params_b
           ]


"""
initialise sensors
"""
l_sensor1 = ColorSensor(INPUT_1)
l_sensor2 = ColorSensor(INPUT_3)
us_sensor = UltrasonicSensor()

#hyperparams
N = 100
dt = 0.00000001
dt = 0.000001
dt_s1 = 0.000001
V = 60 #true hidden state (dist)
V_p = 0 #prior



dist_intervals = np.arange(10, 110, 5)
predictions = {'s1':{} , 's2':{}, 's3':{}}

print("begin\n---------")
for dist in dist_intervals:
    phi1, phi_u1, v1, u1 = run1(N, dt_s1, V, V_p, 1, s1_variance, l_sensor1, g_params[:3])
    print(" prediction 1 sensor:", phi1[len(phi1)-1], "at", dist, "cm")

    phi2, phi_u2, v2, u2 = run2(N, dt, V, V_p, 1, [s1_variance, s2_variance], [l_sensor1, l_sensor2], g_params[:6])
    print(" prediction 2 sensor:", phi2[len(phi2)-1], "at", dist, "cm")

    phi3, phi_u3, v3, u3 = run3(N, dt, dist, V_p, 1, [s1_variance, s2_variance, s3_variance], [l_sensor1, l_sensor2, us_sensor], g_params[:])
    print(" prediction 3 sensor:", phi3[len(phi3)-1], "at", dist, "cm")
    print("\n")
    predictions['s1'][str(dist)] = phi1.tolist()
    predictions['s2'][str(dist)] = phi2.tolist()
    predictions['s3'][str(dist)] = phi3.tolist()

    drive_motors(-50)

with open('phicombined.json', 'w') as outfile:
    json.dump(predictions, outfile)


print("done!")