#!/usr/bin/env python3
import numpy as np
import json
from ev3dev2.sensor import INPUT_1, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
from ev3dev2.display import Display
import time
from ev3dev2.motor import OUTPUT_A, OUTPUT_C, MoveDifferential, SpeedRPM
from ev3dev2.wheel import EV3EducationSetTire


from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints



###############
#KALMAN FILTER#
###############

def objective_func(x, a, b, c):
    """
    objective function for to fit the curve for the generative mapping
    for light sensors
    """
    return (a ** (x-b)) + c

def objective_func_linear(x, a, b):
    """
    for the US sensor
    """
    return (a*x) + b

class wrapped_kf():

    def __init__(self, g_params):
        self.sigmas = MerweScaledSigmaPoints(n=2, alpha=.1, beta=10., kappa=0.)
        self.ukf = UnscentedKalmanFilter(dim_x=2, dim_z=3, dt=1., hx=self.hx, fx=self.fx, points=self.sigmas)
        
        self.s1params_a = g_params[0]
        self.s1params_b = g_params[1]
        self.s1params_c = g_params[2]
        
        self.s2params_a = g_params[3]
        self.s2params_b = g_params[4]
        self.s2params_c = g_params[5]

        self.s3params_a = g_params[6]
        self.s3params_b = g_params[7]

    def setup(self, s1_variance, s2_variance, s3_variance):
        
        self.ukf.P *=50 #state variance (50cm estimate)
        
        self.ukf.R[0, 0] = s1_variance ** 2 # set measurement variance for sensor 1 (top left)
        self.ukf.R[1, 1] = s2_variance ** 2
        self.ukf.R[2, 2] = s3_variance ** 2


    def fx(self, x, dt):
        F = np.array([[1., 0],
                    [0., 1.]]) # state transition matrix
        return F @ x

    def hx(self, x):
        d1 = objective_func(x[0], self.s1params_a, self.s1params_b, self.s1params_c)
        d2 = objective_func(x[0], self.s2params_a, self.s2params_b, self.s2params_c)
        d3 = objective_func_linear(x[0], self.s3params_a, self.s3params_b)
        return [d1, d2, d3]



def run(N, sensor_variances, sensors, g_params, provided_measurements):

    SENSOR_NUM = len(sensors)

    s1_variance = sensor_variances[0]
    s2_variance = sensor_variances[1]
    s3_variance = sensor_variances[2]

    w_ukf = wrapped_kf(g_params)
    w_ukf.setup(s1_variance, s2_variance, s3_variance)


    phi = np.zeros(N) #distance predictions
    u = np.zeros((N, SENSOR_NUM))

    time_log =  np.zeros(N)
    for i in range(0, N):
        #use fed in sensor values rather than recording them internally
        if len(provided_measurements) != 0: 
            u[i, 0] = provided_measurements[0][i]
            u[i, 1] = provided_measurements[1][i]
            u[i, 2] = provided_measurements[2][i]
        else:
            u[i, 0] = sensors[0].ambient_light_intensity #lsensor1
            u[i, 1] = sensors[1].ambient_light_intensity #lsensor2
            u[i, 2] = sensors[2].distance_centimeters #us sensor
        
        z = [u[i, 0], u[i, 1], u[i, 2]]

        start_time =  time.time()
        w_ukf.ukf.predict()
        w_ukf.ukf.update(z)
        end_time = time.time()
        elapsed = end_time - start_time
        time_log[i] = elapsed
        

        phi[i] = w_ukf.ukf.x[0]
        print('Epoch \r',i+1, '/', N, end="")


    logs = {
        'phi':phi,
        'u': u,
        'time':time_log
    }
    return logs