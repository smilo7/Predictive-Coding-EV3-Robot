#!/usr/bin/env python3
from models.one_sensor import run as run1
from models.two_sensor import run as run2
from models.three_sensor import run as run3
from models.unscented_KF import run as run_ukf
from models.three_sensor import robot_brain as robot_brain_reusable

from models.shared_functions import read_data_from_file_json, drive_motors

import csv
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


############################
#          PRINT UTIL      #
############################
def print_prediction(phis, dist, model_name):
    # get last prediction
    pred = phis[len(phis)-1]
    print(" prediction ", model_name, ':', pred, "at", dist, "cm")


def save_log(filename, log, directory='logs/'):
    """
    save logs in dictionary format
    """
    PATH = directory+filename+'.csv'
    with open(PATH, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, log.keys())
        w.writeheader()
        w.writerow(log)
    print('log written to: '+PATH)



############################
#          SENSORS         #
############################
"""
initialise sensors
"""
l_sensor1 = ColorSensor(INPUT_1)
l_sensor2 = ColorSensor(INPUT_3)
us_sensor = UltrasonicSensor()

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

def mean_of_list(data):
    return sum(data) / len(data)


############################
#           RUN            #
############################

#hyperparams
dt = 0.000001
V = 60 #true hidden state (dist)
V_p = 0 #prior

# variance learning rate
lr_sigmas = [0.1, 0.01, 0.001, 0.0001]

# whether to giev the same measurements to each 
# robot_brain/filter or have them do it internally

measurements_together = True
measurement_log = []


PC1_timings = []
PC2_timings = []
PC3_timings = []
UKF1_timings = []
UKF2_timings = []
UKF3_timings = []


N_steps = [1, 10, 100, 1000, 10000, 100000]
N_steps = [100, 100, 100, 100, 100 , 100, 100, 100, 100, 100]
#dist_intervals = np.arange(20, 85, 5)

predictions = {'s1':{} , 's2':{}, 's3':{}, 'kf2':{}, 's2_multi':{}, 's3_learning':{}}

dist=V

print("begin\n---------")
for N in N_steps:

    provided_measurements = []
    if measurements_together:
        #                        s1   s2  us
        provided_measurements = [[], [], []]
        #get N measurements
        for i in range(0, N):
            provided_measurements[0].append(l_sensor1.ambient_light_intensity)
            provided_measurements[1].append(l_sensor2.ambient_light_intensity)
            provided_measurements[2].append(us_sensor.distance_centimeters)

    measurement_log.append(provided_measurements)
    print(N ,'input size')

    logs_1 = run1(N, dt, V, V_p, 1, s1_variance, l_sensor1, g_params[:3], provided_measurements)    
    print_prediction(logs_1['phi'], dist, '1 sensor')
    PC1_timings.append(sum(logs_1['time']))

    logs_2 = run2(N, dt, V, V_p, 1, [s1_variance, s2_variance], [l_sensor1, l_sensor2], g_params[:], provided_measurements, multisensory=False)
    print_prediction(logs_2['phi'], dist, '2 sensor')
    PC2_timings.append(sum(logs_2['time']))

    logs_3 = run3(N, dt, dist, V_p, 1, [s1_variance, s2_variance, s3_variance], [l_sensor1, l_sensor2, us_sensor], g_params[:], provided_measurements)
    print_prediction(logs_3['phi'], dist, '3 sensor')
    PC3_timings.append(sum(logs_3['time']))
    

    logs_ukf = run_ukf(N, [s1_variance, s2_variance, s3_variance],  [l_sensor1, l_sensor2, us_sensor], g_params[:], provided_measurements, num_sensors=1)
    print_prediction(logs_ukf['phi'], dist, 'UKF')
    print('ukf2', sum(logs_ukf['time']))
    UKF1_timings.append(sum(logs_ukf['time']))

    logs_ukf = run_ukf(N, [s1_variance, s2_variance, s3_variance],  [l_sensor1, l_sensor2, us_sensor], g_params[:], provided_measurements, num_sensors=2)
    print_prediction(logs_ukf['phi'], dist, 'UKF')
    print('ukf2', sum(logs_ukf['time']))
    UKF2_timings.append(sum(logs_ukf['time']))

    logs_ukf = run_ukf(N, [s1_variance, s2_variance, s3_variance],  [l_sensor1, l_sensor2, us_sensor], g_params[:], provided_measurements, num_sensors=3)
    print_prediction(logs_ukf['phi'], dist, 'UKF')
    print('ukf3', sum(logs_ukf['time']))
    UKF3_timings.append(sum(logs_ukf['time']))



    print(PC1_timings)
    print(PC2_timings)
    print(PC3_timings)
    print(UKF1_timings)
    print(UKF2_timings)
    print(UKF3_timings)

    print(mean_of_list(PC1_timings))
    print(mean_of_list(PC2_timings))
    print(mean_of_list(PC3_timings))
    print(mean_of_list(UKF1_timings))
    print(mean_of_list(UKF2_timings))
    print(mean_of_list(UKF3_timings))

with open("PC_timings.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(PC_timings)

with open("UKF_timings.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(UKF_timings)