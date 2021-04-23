#!/usr/bin/env python3
import numpy as np
import json
from ev3dev2.sensor import INPUT_1, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
from ev3dev2.display import Display
import time
from ev3dev2.motor import OUTPUT_A, OUTPUT_C, MoveDifferential, SpeedRPM
from ev3dev2.wheel import EV3EducationSetTire


class robot_brain:
    """
    This class deals with the neuronal working of the robots brain state(s)
    """
    def __init__(self, dt, V_p, Sigma_p, Sigma_u, g_params, multisensory):
        self.dt = dt
        self.phi = V_p #phi is the current best guess, we initialise it to the "hierachical prior"

        self.eps_p = 0 #prediction error on distance the hidden state delta epsilon_p in bogacz
        self.eps_u1 = 0 #prediction error on sensory input delta epsilon_u in bogacz
        self.eps_u2 = 0
        self.Sigma_p = Sigma_p #variance of distance (hidden state)
        self.Sigma_u = Sigma_u #variance of sensory input
        
        self.F = 0 #Free energy!

        self.s1params_a = g_params[0]
        self.s1params_b = g_params[1]
        self.s1params_c = g_params[2]
        
        self.s2params_a = g_params[3]
        self.s2params_b = g_params[4]
        self.s2params_c = g_params[5]

        self.s3params_a = g_params[6]
        self.s3params_b = g_params[7]

        self.multisensory = multisensory
    
    def g(self, phi, sensor_num):
        #generative model
        if sensor_num == 0:
            return self.g_true_l1(phi)
        elif sensor_num ==1:
            if self.multisensory:
                return self.g_true_l3(phi)
            else:
                return self.g_true_l2(phi)
    
    def g_deriv(self, phi, sensor_num):
        if sensor_num == 0:
            return self.g_true_deriv_l1(phi)
        elif sensor_num == 1:
            if self.multisensory: # use ultrasonic sensor mapping
                return self.g_true_deriv_l3(phi)
            else:
                return self.g_true_deriv_l2(phi)


    def g_true_l1(self, x):
        """
        generative function for light sensor 1
        """
        a = self.s1params_a
        b = self.s1params_b
        c = self.s1params_c
        return (a ** (x-b)) + c

    def g_true_l2(self, x):
        """
        generative function for light sensor 3
        """
        a = self.s2params_a
        b = self.s2params_b
        c = self.s2params_c
        return (a ** (x-b)) + c

    def g_true_l3(self, x):
        """
        generative function for light sensor 3
        """
        a = self.s3params_a
        b = self.s3params_b
        return (a*x) + b

    def g_true_deriv_l1(self, x):
        a = self.s1params_a
        b = self.s1params_b
        return (x-b) * (a**(x-b-1))

    def g_true_deriv_l2(self, x):
        a = self.s2params_a
        b = self.s2params_b
        return (x-b) * (a**(x-b-1))

    #ultrasonic sensor gmap for multisensory option
    def g_true_deriv_l3(self, x):
        a = self.s3params_a
        return a

    def calc_free_energy(self, V_p, u):
        """
        Calculates the free energy at the current step
        inputs
        V_p - hierachical prior
        u - sensory input
        """
        e_p = (self.phi - V_p)
        #e_u = (u[0] - self.g(self.phi, 0))
        e_u1 = (u[0] - self.g(self.phi, 0)) * (1/self.Sigma_u[0])
        e_u2 = (u[1] - self.g(self.phi, 1)) * (1/self.Sigma_u[1])
        F = (e_p**2 / self.Sigma_p + e_u1**2 + self.Sigma_u[0] + e_u2 +self.Sigma_u[1] + np.log(self.Sigma_p * self.Sigma_u[0] * self.Sigma_u[1]) ) / 2
        return F

    def inference(self, V_p, u):
        """
        #i - timestep
        mu_p - prior guess of hidden state
        u - sensory input
        """

        lr = 1
        dt = self.dt
        #page 4 of bogacz for the maths

        #Prediction errors
        e_p = (self.phi - V_p) * (1/self.Sigma_p) #current best guess of distance MINUS prior best guess (epsilon_p in bogacz)
        #print("here", self.g(self.phi, 0), self.g(self.phi, 1))
        e_u1 = (u[0] - self.g(self.phi, 0)) * (1/self.Sigma_u[0]) #sensory reading - the light level of the generative model (the difference between sensory input and the inner model) (epsilon_u in bogacz)
        e_u2 = (u[1] - self.g(self.phi, 1)) * (1/self.Sigma_u[1])

        #computation carried out in nodes to compute prediction errors.
        self.eps_p = self.eps_p + dt * lr * (e_p- self.Sigma_p * self.eps_p) # moves towards mean of prior
        self.eps_u1 = self.eps_u1 + dt * lr * (e_u1 - self.Sigma_u[0] * self.eps_u1) # moves according to sensory input
        self.eps_u2 = self.eps_u2 + dt * lr * (e_u2 - self.Sigma_u[1] * self.eps_u2)

        #self.phi = self.phi + dt * lr * ( - self.eps_p + self.eps_u * self.g_deriv(self.phi)) 
        #self.phi = self.phi + dt * lr * (- self.eps_p + self.eps_u1 * self.g_deriv(self.phi, 0) + self.eps_u2 * self.g_deriv(self.phi, 1))
        self.phi = self.phi + dt * lr * (
                                        - e_p 
                                        + e_u1 * self.g_deriv(self.phi, 0) 
                                        + e_u2 * self.g_deriv(self.phi, 1)
                                        )

        #self.F = self.calc_free_energy(u)

        phi_u = self.g(self.phi, 0) #sensory_predictions, based on feeding the current prediction of the hideen state to the generative model

        return self.phi, phi_u


def run(N, dt, V, V_p, Sigma_p, Sigma_u, l_sensors, g_params, provided_measurements, multisensory):
    """
    iterate the robot brain through time
    N- Number of steps to simulate
    V - actual distance (for use with generating sensory input in this simulation)
    V_p - Hidden state prior
    Sigma_p - Estimated variance of hidden state
    Sigma_u - estimated variance of the sensory input
    l_sensor - light sensor object
    """

    robot = robot_brain(dt, V_p, Sigma_p, Sigma_u, g_params, multisensory=multisensory)

    SENSOR_NUM = len(l_sensors)#2
    #inititalise arrays for recording results
    phi = np.zeros(N)
    phi_u = np.zeros(N) # prediction of sensory input
    F = np.zeros(N) #free energy
    v = np.zeros(N) # true distance (hidden state)
    u = np.zeros((N, SENSOR_NUM)) #sensory input

    time_log = np.zeros(N)
    #phi[0] = V_p #in
    for i in range(0, N):
        v[i] = V
        #use fed in sensor values rather than recording them internally
        if len(provided_measurements) != 0: 
            u[i, 0] = provided_measurements[0][i]
            u[i, 1] = provided_measurements[1][i]
        else:
            if multisensory:
                u[i, 0] = l_sensors[0].ambient_light_intensity #take sensor reading
                u[i, 1] = l_sensors[1].distance_centimeters # take distance measurement rather than light intensity
            else:
                u[i, 0] = l_sensors[0].ambient_light_intensity #take sensor reading
                u[i, 1] = l_sensors[1].ambient_light_intensity

        #u[i] = g_true(V) #sensory input given as true generative process generating sensory input
        
        start_time = time.time()
        phi[i], phi_u[i] = robot.inference(V_p, u[i])#do inference at the current timestep with the previous hierachical prior, and current sensory input
        end_time = time.time()
        elapsed = end_time - start_time
        time_log[i]= elapsed
        print('Epoch \r',i+1, '/', N, end="")

    
    
    logs = {
        'phi':phi,
        'phi_u':phi_u,
        'F': F,
        'v': v,
        'u': u,
        'time':time_log
    }

    return logs