#!/usr/bin/env python3
import numpy as np
import math
import json
from ev3dev2.sensor import INPUT_1, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
from ev3dev2.display import Display
import time
from ev3dev2.motor import OUTPUT_A, OUTPUT_C, MoveDifferential, SpeedRPM
from ev3dev2.wheel import EV3EducationSetTire

from models.shared_functions import read_data_from_file_json, drive_motors

from collections import namedtuple #for guassians

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

################################################################################

################################################################################

l_sensor1 = ColorSensor(INPUT_1)

def g_true_l1(x):
    """
    generative function for light sensor 1
    """
    a = s1params_a
    b = s1params_b
    c = s1params_c
    return (a ** (x-b)) + c

def g_true_deriv_l1(x):
    a = s1params_a
    b = s1params_b
    return (x-b) * (a**(x-b-1))

def convert_light_to_dist(y):
    a = s1params_a
    b = s1params_b
    c = s1params_c
    return (math.log(y) + math.log(1-(c/y)) ) - math.log(a) + b


################################################################################

################################################################################
gaussian = namedtuple('Gaussian', ['mean', 'variance'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])

# kalman filter
def predict(posterior, movement):
    x, P = posterior # mean and variance of posterior
    dx, Q = movement # mean and variance of movement
    
    x = x + dx
    P = P + Q
    return gaussian(x, P)

def update(prior, measurement):
    x, P = prior        # mean and variance of prior
    z, R = measurement  # mean and variance of measurement
    
    y = z - x        # residual. difference between mean of measurement and mean of prior. (kind of like prediction error...)
    K = P / (P + R)  # Kalman gain as a ratio between uncertainty in prior and measurement

    x = x + K*y      # posterior, calculate the posterior by 
    P = (1 - K) * P  # posterior variance
    return gaussian(x, P)


def run(N, sensor):
    position_change = 0 # we expect the position of the robot to stay the same (as it is stationary)
    
    sensor_std = math.sqrt(s1_variance) # light reading standard deviation, check manufacturer for standard deviation of noise
    process_variance = s1_variance #expected change in light reading over each step

    print('sensor_std', sensor_std, 'process_variance', process_variance)

    x = gaussian(0., 1000.) #guassian for sensor value
    process_model = gaussian(0., process_variance)

    ps = []
    estimates = []
    z = np.zeros(N) #actual sensor readings

    for i in range(0, N):
        z[i] = sensor.ambient_light_intensity #get sensor reading


        prior = predict(x, process_model) #predict
        x = update(prior, gaussian(z[i], sensor_std)) #update

        estimates.append(x.mean) #log 
        ps.append(x.variance)

    return estimates, ps, z

estimates, ps, z =  run(N=50, sensor=l_sensor1)

print("estimates of actual value of sensory input", estimates)

print("done")

distances = [convert_light_to_dist(each) for each in estimates]
#distances = g_true_l1(estimates)
print(z)
print("dist based on z", convert_light_to_dist(z[len(z)-1]))
print("dist based on predictions", convert_light_to_dist(estimates[len(estimates)-1]))

print(distances)