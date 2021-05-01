#!/usr/bin/env python3
from models.one_sensor import run as run1
from models.two_sensor import run as run2
from models.three_sensor import run as run3
from models.unscented_KF import run as run_ukf

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


"""
Read in the lookup table for converting predicted light intensities
for the kalman filter
"""
lookup_tablejson = read_data_from_file_json('lookup_table.json')

# convert keys back into ints
lookup_table = {}
for k, v in lookup_tablejson.items():
    lookup_table[k] = {int(key): int(value) for key, value in v.items()}

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




############################
#           RUN            #
############################

#hyperparams
N = 100
dt = 0.00000001
dt = 0.000001
dt_s1 = 0.000001
V = 60 #true hidden state (dist)
V_p = 0 #prior
#dt = 0.00001
# whether to giev the same measurements to each 
# robot_brain/filter or have them do it internally
measurements_together = True
measurement_log = []

dist_intervals = np.arange(10, 115, 5)
#dist_intervals = np.arange(20, 85, 5)

predictions = {'s1':{} , 's2':{}, 's3':{}, 'kf2':{}, 's2_multi':{}, 's3_learning':{}}

print("begin\n---------")
for dist in dist_intervals:

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
        #provided_measurements[0] = l_sensor1.ambient_light_intensity #lsensor1
        #provided_measurements[1] = l_sensor2.ambient_light_intensity
        #provided_measurements[2] = us_sensor.distance_centimeters
    #if measurements_together:
    #    print('measurements', provided_measurements)

    logs_1 = run1(N, dt, V, V_p, 1, s1_variance, l_sensor1, g_params[:3], provided_measurements)    
    print_prediction(logs_1['phi'], dist, '1 sensor')

    logs_2 = run2(N, dt, V, V_p, 1, [s1_variance, s2_variance], [l_sensor1, l_sensor2], g_params[:], provided_measurements, multisensory=False)
    print_prediction(logs_2['phi'], dist, '2 sensor')

    logs_3 = run3(N, dt, dist, V_p, 1, [s1_variance, s2_variance, s3_variance], [l_sensor1, l_sensor2, us_sensor], g_params[:], provided_measurements)
    print_prediction(logs_3['phi'], dist, '3 sensor')

    logs_3learning = run3(N, dt, dist, V_p, 1, [s1_variance, s2_variance, s3_variance], [l_sensor1, l_sensor2, us_sensor], g_params[:], provided_measurements, learn_variances=True)
    print_prediction(logs_3learning['phi'], dist, '3 sensor with learning')

    # make sure US sensor is in second slot in the list for muyltisensor True
    logs_2multisensor = run2(N, dt, V, V_p, 1, [s1_variance, s3_variance], [l_sensor1, us_sensor], g_params[:], provided_measurements, multisensory=True)
    print_prediction(logs_2multisensor['phi'], dist, '2 sensor multisensor')

    logs_ukf = run_ukf(N, [s1_variance, s2_variance, s3_variance],  [l_sensor1, l_sensor2, us_sensor], g_params[:], provided_measurements)
    print_prediction(logs_ukf['phi'], dist, 'UKF')

    #x_kf2 = run_kf2(N, [s1_variance, s2_variance], [l_sensor1, l_sensor2], lookup_table)
    #print(" prediction 2 sensor KF:", x_kf2[len(x_kf2)-1], "at", dist, "cm")

    print("\n")
    
    predictions['s1'][str(dist)] = logs_1['phi'].tolist()
    predictions['s2'][str(dist)] = logs_2['phi'].tolist()
    predictions['s3'][str(dist)] = logs_3['phi'].tolist()
    predictions['kf2'][str(dist)] = logs_ukf['phi'].tolist()
    predictions['s2_multi'][str(dist)] = logs_2multisensor['phi'].tolist()
    predictions['s3_learning'][str(dist)] = logs_3learning['phi'].tolist()
   

    drive_motors(-50)

with open('phicombined_with_kalman_100iter10.json', 'w') as outfile:
    json.dump(predictions, outfile)

with open("sensor_readings_7.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(measurement_log)

# save full log for luck
save_log('1sensor', logs_1)
save_log('2sensor', logs_2)
save_log('3sensor', logs_3)
save_log('2sensor_multi', logs_2multisensor)
save_log('ukf', logs_ukf)
save_log('3sensor_learning', logs_3learning)


print("done!")