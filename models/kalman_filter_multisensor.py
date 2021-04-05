#!/usr/bin/env python3
import numpy as np
import json
from ev3dev2.sensor import INPUT_1, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
from ev3dev2.display import Display
import time
from ev3dev2.motor import OUTPUT_A, OUTPUT_C, MoveDifferential, SpeedRPM
from ev3dev2.wheel import EV3EducationSetTire


from scipy.linalg import inv


class kalman_filter():
    def __init__(self, x, P, F, Q, H, R):
        self.x = x
        self.P = P
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q


    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ inv(S)
        self.y = z - self.H @ self.x
        self.x += self.K @ self.y
        self.P = self.P - self.K @ self.H @ self.P



def make_filter_2sensors(s1_variance, s2_variance, state_variance):

    x = np.array([[0.], 
                  [0.]]) # initialise the state pos 0 and velocity 0

    P = np.array([[1., 0],
                  [0., 1.]]) # P state variance

    P *= state_variance #multiply diagonal rows by state_variance
    
    F = np.array([[1., 0],
                  [0., 1.]]) # state transition matrix

    Q = np.array([[1., 0],
                  [0., 1.]]) # process model covariance
    
    H = np.array([[1., 0.],
                  [1., 0.]]) #  measurement function (converts from prediction to measurement space)

    R = np.array([[1., 0],
                  [0., 1.]])
    R[0, 0] = s1_variance ** 2 # set measurement variance for sensor 1 (top left)
    R[1, 1] = s2_variance ** 2 # set measurement variance for sensor 2 (bottom right)

    return kalman_filter(x, P, F, Q, H, R)


def convert(sensor_variances, lookup_table, x):
    """
    convert using lookup table
    """
    l_reading = int(x) #round to int for now

    dist_s1 = lookup_table['s1'][l_reading]
    dist_s2 = lookup_table['s2'][l_reading]
   
    sv = sensor_variances
    
    fused_dist = ( (dist_s1 * sv[0]**-1) + (dist_s2 * sv[1]**-1) ) / (sv[0]**-1 + sv[1]**-1)

    return fused_dist


def run(N, sensor_variances, sensors, lookup_table):
    
    
    kf = make_filter_2sensors(sensor_variances[0], sensor_variances[1], 50)

    x_log = np.zeros(N)

    for i in range(0, N):
        s1 = sensors[0].ambient_light_intensity
        s2 = sensors[1].ambient_light_intensity

        z = np.array([[s1], [s2]]) # measurement

        kf.predict()
        kf.update(z)

        #save state to look at later
        x_log[i] = kf.x[0][0] # log just the position
    
    #convert to distance values using the lookup table
    x_log_dists = [convert(sensor_variances, lookup_table, x) for x in x_log]

    return x_log_dists