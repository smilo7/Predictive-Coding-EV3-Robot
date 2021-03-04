#!/usr/bin/env python3
import numpy as np
import json
from ev3dev2.sensor import INPUT_1, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
from ev3dev2.display import Display
import time


class robot_brain():
    """
    This class deals with the neuronal working of the robots brain state(s)
    """
    def __init__(self, dt, V_p, Sigma_p, Sigma_u):
        self.dt = dt
        self.phi = V_p #phi is the current best guess, we initialise it to the "hierachical prior"

        self.eps_p = 0 #prediction error on distance the hidden state delta epsilon_p in bogacz
        self.eps_u = 0 #prediction error on sensory input delta epsilon_u in bogacz
        self.Sigma_p = Sigma_p #variance of distance (hidden state)
        self.Sigma_u = Sigma_u #variance of sensory input
        
        self.F = 0 #Free energy!
    
    def g(self, phi):
        #generative model
        return g_true(phi)
    
    def g_deriv(self, phi):
        return g_true_deriv(phi)


    def calc_free_energy(self, V_p, u):
        """
        Calculates the free energy at the current step
        inputs
        V_p - hierachical prior
        u - sensory input
        """
        #recalculate prediction errors because prediction of hidden state might have been updated
        e_p = (self.phi - V_p)
        e_u = (u - self.g(self.phi))
        F = (e_u**2 / self.Sigma_u + e_u**2 + self.Sigma_p + np.log(self.Sigma_p * Sigma_u)) / 2
        return F

    def inference(self, V_p, u):
        """
        #i - timestep
        mu_p - prior guess of hidden state
        u - sensory input
        """

        lr = 0.1

        #page 4 of bogacz for the maths

        #Prediction errors
        # I THINK HERE IS WHERE YOU DO THE PRECISSION CALCULATIONS. AS 1/variance is precision. sigmas are the varainces :) MAYBE?? not sure
        e_p = (self.phi - V_p) * (1/self.Sigma_p)#/ self.Sigma_p #current best guess of distance MINUS prior best guess (epsilon_p in bogacz)
        e_u = (u - self.g(self.phi) ) * (1/self.Sigma_u) #/ Sigma_u #sensory reading - the light level of the generative model (the difference between sensory input and the inner model) (epsilon_u in bogacz)

        #computation carried out in nodes to compute prediction errors.
        self.eps_p = self.eps_p + dt * lr * (e_p- self.Sigma_p * self.eps_p) # moves towards mean of prior
        self.eps_u = self.eps_u + dt * lr * (e_u - self.Sigma_u * self.eps_u) # moves according to sensory input
        
        #self.phi = self.phi + dt * lr * ( - e_p + e_u * self.g_deriv(self.phi)) #update best guess with our new prediction! TODO probably throw in a learning rate in here.
        
        self.phi = self.phi + dt * lr * ( - e_p + e_u * self.g_deriv(self.phi))
        
        #self.F = self.calc_free_energy(u)

        phi_u = self.g(self.phi) #sensory_predictions, based on feeding the current prediction of the hideen state to the generative model

        return self.phi, phi_u


def run(N, dt, V, V_p, Sigma_p, Sigma_u, l_sensor):
    """
    iterate the robot brain through time
    N- Number of steps to simulate
    V - actual distance (for use with generating sensory input in this simulation)
    V_p - Hidden state prior
    Sigma_p - Estimated variance of hidden state
    Sigma_u - estimated variance of the sensory input
    l_sensor - light sensor object
    """

    robot = robot_brain(dt, V_p, Sigma_p, Sigma_u)

    #inititalise arrays for recording results
    phi = np.zeros(N)
    phi_u = np.zeros(N) # prediction of sensory input
    F = np.zeros(N) #free energy
    v = np.zeros(N) # true distance (hidden state)
    u = np.zeros(N) #sensory input

    #phi[0] = V_p #in

    for i in range(0, N):
        v[i] = V
        u[i] = l_sensor.ambient_light_intensity #take sensor reading
        
        #u[i] = g_true(V) #sensory input given as true generative process generating sensory input
        phi[i], phi_u[i] = robot.inference(V_p, u[i]) #do inference at the current timestep with the previous hierachical prior, and current sensory input
        print('Epoch \r',i+1, '/', N, end="")
    
    #print("light readings", u)
    return phi, phi_u, v , u


def read_data_from_file_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def g_true(x):
    a = s1params_a
    b = s1params_b
    c = s1params_c
    return (a ** (x-b)) + c

def g_true_deriv(x):
    a = s1params_a
    b = s1params_b
    return (x-b) * (a**(x-b-1))


"""
calc mean variance from the recordings
"""
data = read_data_from_file_json('data_out.json')

s1_variance = sum(data['s1']['variances'])/len(data['s1']['variances'])
print("variances s1:", s1_variance)
"""
read params from file
"""
params = read_data_from_file_json('genmap_params.json')

#assign params to globals
s1params_a, s1params_b, s1params_c = params['s1']['a'], params['s1']['b'], params['s1']['c']

"""
initialise sensors
"""
l_sensor1 = ColorSensor(INPUT_1)

#hyperparams
N = 30000
dt = 0.0025
N = 10000
dt = 0.0001
V = 60 #true hidden state (dist)
V_p = 0 #prior

"""
run the robots inference program
"""
phi, phi_u, v, u = run(N, dt, V, V_p, 1, s1_variance, l_sensor1)
print("final predic", phi[len(phi)-1])

"""
write out the results to file
"""
dataa = {'phi':phi.tolist()} 
with open('phi1sensor.json', 'w') as outfile:
    json.dump(dataa, outfile)
#write_results_to_csv('inference2.csv', phi, phi_u[0], phi_u[1],v, u) #write results out to csv file
print(phi, phi_u)