#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import math
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
from ev3dev2.display import Display
import time


def read_data_from_file(filename):
    return np.genfromtxt(filename, delimiter=',')

def objective_func(x, a, b, c):
    return (a ** (x-b)) + c

def fit_function_to_data(distance, light):
    """
    fits a function to the data and then returns the parameters
    """
    params, _ = curve_fit(objective_func, distance, light)
    a, b, c = params
    print("y = (",a,"(^x-",b,")) +",c)
    x_line = np.arange(min(distance), max(distance))
    y_line = objective_func(x_line, a, b, c)
    return a, b, c


def g_gp(x,v):
    a = params_a
    b = params_b
    c = params_c
    return (a ** (x-b)) + c

def dg_gp(x):
    a = params_a
    b = params_b
    return (x-b) * (a**(x-b-1))



class pp_unit():
    """
        Class that constructs a group of neurons that perform Active Inference for one hidden state, one sensory input, one prior
        In neurology it could eg represent a (micro) column
        
        Version 0.3 Alternative approach where the prediction error is cast as a motion 
    """
    def __init__(self, dt, mu_v, Sigma_w, Sigma_z, a_mu):   
        self.dt = dt    # integration step
        self.mu_x = mu_v   # initializing the best guess of hidden state by the hierarchical prior
        self.F = 0      # Free Energy
        self.eps_x = 0  # delta epsilon_x, prediction error on hidden state
        self.eps_y = 0  # delta epsilon_y, prediction error on sensory measurement
        self.Sigma_w = Sigma_w #Estimated variance of the hidden state 
        self.Sigma_z = Sigma_z # Estimated variance of the sensory observation 
        self.alpha_mu = a_mu # Learning rate of the gradient descent mu (hidden state)
    
    def g(self,x, v):
        """
            equation of sensory mapping of the generative model: g(x) at point x 
            Given as input for this example equal to the true generative process g_gp(x)
            x is the current best guess of the hidden state (distance) 
        """
        
        #return 43.11
        return g_gp(x,v)
    
    def dg(self, x):
        """
            Partial derivative of the equation of sensory mapping of the generative model towards x: g'(x) at point x 
            Given as input for this example equal to the true derivative of generative process dg_gp(x)
        """
        
        #return -0.411
        return dg_gp(x)
    
    def f(self,x,v):
        """
            equation of motion of the generative model: f(x) at point x 
            Given as input for this example equal to the prior belief v
        """
        return v
    
    # def df(self,x): Derivative of the equation of motion of the generative model: f'(x) at point x
    # not needed in this example 

    
    def inference_step (self, i, mu_v, y):
        """
        Perceptual inference    

        INPUTS:
            i       - tic, timestamp
            mu_v    - Hierarchical prior input signal (mean) at timestamp
            y       - sensory input signal at timestamp

        INTERNAL:
            mu      - Belief or hidden state estimation

        """
       
        # Calculate the prediction errors
        e_x = self.mu_x - self.f(self.mu_x, mu_v)  # prediction error hidden state
        e_y = y - self.g(self.mu_x, mu_v) #prediction error sensory observation
        
        # motion of prediction error hidden state
        self.eps_x = self.eps_x + dt * self.alpha_mu * (e_x - self.Sigma_w * self.eps_x)
        # motion of prediction error sensory observation
        self.eps_y = self.eps_y + dt * self.alpha_mu * (e_y - self.Sigma_z * self.eps_y)
        # motion of hidden state mu_x 
        self.mu_x = self.mu_x + dt * - self.alpha_mu * (self.eps_x - self.dg(self.mu_x) * self.eps_y)
        
        # Calculate Free Energy to report out
        # Recalculate the prediction errors because hidden state has been updated, could leave it out for performance reasons
        e_x = self.mu_x - self.f(self.mu_x, mu_v)  # prediction error hidden state
        e_y = y - self.g(self.mu_x, mu_v) #prediction error sensory observation
        # Calculate Free Energy
        self.F = 0.5 * (e_x**2 / self.Sigma_w + e_y**2 / self.Sigma_z + np.log(self.Sigma_w * self.Sigma_z))
        
        return self.F, self.mu_x , self.g(self.mu_x,0)


def run(mu_v, Sigma_w, Sigma_z, a_mu, l_sensor):
    """
    Basic simplist example perceptual inference    

    INPUTS:
        mu_v     - Robot prior belief/hypotheses of the hidden state
        Sigma_w  - Estimated variance of the hidden state 
        Sigma_z  - Estimated variance of the sensory observation  
        a_mu     - Learning rate for mu
    """
    N = 1000
    # Init tracking
    mu_x = np.zeros(N) # Belief or estimation of hidden state 
    F = np.zeros(N) # Free Energy of AI neuron
    mu_y = np.zeros(N) # Belief or prediction of sensory signal 
    x = np.zeros(N) # True hidden state
    y = np.zeros(N) # Sensory signal as input to AI neuron

    robot_brain = pp_unit(dt, mu_v, Sigma_w, Sigma_z, a_mu) #make pp object
    
    

    start_time = time.time()
    for i in np.arange(1, N):
        #Active inference
        y[i] = l_sensor.ambient_light_intensity #take sensor reading
        print('ligght reading', y[i])
        F[i], mu_x[i], mu_y[i] = robot_brain.inference_step(i, mu_v, y[i])


    t_elapsed = time.time() - start_time

    print("Elapsed Time", t_elapsed, "sec")
    return F, mu_x, mu_y, x, y


def write_results_to_csv(filename, F1, mu_x, mu_y, x, y):
    data = {'F1':F1, 'mu_x':mu_x, 'mu_y':mu_y, 'x':x, 'y':y}    
    df = pd.DataFrame(data)
    df.to_csv(filename)

s_light = ColorSensor() #initialise the color sensor

dt = 0.005 #timestep
actual_dist = 60
dist_prior = 40 #say it thinks it is 25cm away to start with. this is its belief

#find the params based on data for the generative function g_gp(x)
light_data = read_data_from_file('data.csv')
distances = np.arange(0, 100, 5)
print("file contents", light_data, type(light_data))

params_a, params_b, params_c = fit_function_to_data(distances, light_data)

F1, mu_x, mu_y, x, y = run(mu_v=dist_prior, Sigma_w=1, Sigma_z=1, a_mu=1, l_sensor=s_light)

write_results_to_csv('inference.csv', F1, mu_x, mu_y, x, y) #write results out to csv file

print(F1, mu_x, mu_y, x, y)