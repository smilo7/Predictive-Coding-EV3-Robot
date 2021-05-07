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


print(g_params)


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


def get_last(data):
    """
    return last value from a list
    """
    return data[len(data)-1]

############################
#           RUN            #
############################

#hyperparams
N = 1000 #number of inference steps
#dt = 0.00000001
dt = 0.000001
#dt_s1 = 0.000001
V = 60 #true hidden state (dist)
V_p = 0 #prior

# variance values
s1_start_variance = 1
s2_start_variance = 1
s3_start_variance = 1


# variance learning rate
lr_sigmas = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
learning_rates = {'good_variances':[], 'no_lr_bad_variances':[], 1:[], 0.1:[], 0.01:[], 0.001:[], 0.0001:[], 0.00001:[]} #predictions for each learning rate
sigma_preds = {'good_variances':[], 'no_lr_bad_variances':[], 1:[], 0.1:[], 0.01:[], 0.001:[], 0.0001:[], 0.00001:[]} # predictions for sigma (sensory variance)

# whether to give the same measurements to each 
# robot_brain/filter or have them do it internally

measurements_together = True
measurement_log = []

#dist_intervals = np.arange(10, 115, 5)
#dist_intervals = np.arange(20, 85, 5)

predictions = {'s3_learning':{} , 's3':{}, 's3_bad_start':{}}

#robot_brain_reuse = robot_brain_reusable(dt=dt, V_p=0, Sigma_p=1, Sigma_u=)

# we can test just one distance
dist = 30 #cm
trials = 10
for trial in range(0, trials):
    print("begin\n---------")
    #for dist in dist_intervals:

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


    #calcualte the variance of the sensory readings of this instance
    print('s1 variance', calc_variance(provided_measurements[0]))
    print('s2 variance', calc_variance(provided_measurements[1]))
    print('s3 variance', calc_variance(provided_measurements[2]))

    logs_3 = run3(N, dt, dist, V_p, 1, [s1_variance, s2_variance, s3_variance], [l_sensor1, l_sensor2, us_sensor], g_params[:], provided_measurements)
    print_prediction(logs_3['phi'], dist, '3 sensor')
    #print(sum(logs_3['time']),  'time 3 sensor')
    learning_rates['good_variances'].append(get_last(logs_3['phi']))
    sigma_preds['good_variances'].append(get_last(logs_3['Sigma_u']))

    # no learning with bad starting variance
    logs_3_bad_start = run3(N, dt, dist, V_p, 1, [s2_start_variance, s2_start_variance, s2_start_variance], [l_sensor1, l_sensor2, us_sensor], g_params[:], provided_measurements)
    print_prediction(logs_3_bad_start['phi'], dist, '3 sensor bad start')
    learning_rates['no_lr_bad_variances'].append(get_last(logs_3_bad_start['phi']))
    sigma_preds['no_lr_bad_variances'].append(get_last(logs_3_bad_start['Sigma_u']))

    #for key, value in learning_rates.items():
    for key in lr_sigmas:
        logs_3learning = run3(N, dt, dist, V_p, 1, [s1_start_variance, s2_start_variance, s3_start_variance], [l_sensor1, l_sensor2, us_sensor], g_params[:], provided_measurements, learn_variances=True, lr_sigmas=key)
        
        #append the last prediction to the relevant learning rate
        learning_rates[key].append(get_last(logs_3learning['phi']))
        sigma_preds[key].append(get_last(logs_3learning['Sigma_u'])) # same keys so can stick in this loop

        print_prediction(logs_3learning['phi'], dist, '3 sensor with learning'+str(key))
        #print(logs_3learning['Sigma_u'], dist, '3 sensor with learning')



    print("\n")

    predictions['s3'][str(dist)] = logs_3['phi'].tolist()
    #predictions['kf2'][str(dist)] = logs_ukf['phi'].tolist()
    predictions['s3_learning'][str(dist)] = logs_3learning['phi'].tolist()


    #drive_motors(-50) # drive the motor backwards

print('done predicting')



print(learning_rates)
print(sigma_preds)

with open('learning_variances.json', 'w') as outfile:
    json.dump(predictions, outfile)

with open("sensor_readings_7.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(measurement_log)

# save full log for luck
save_log('3sensor', logs_3)
#save_log('ukf', logs_ukf)
save_log('3sensor_learning', logs_3learning)


print("done!")