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
        self.eps_u1 = 0 #prediction error on sensory input delta epsilon_u in bogacz
        self.eps_u2 = 0
        self.eps_u3 = 0
        self.Sigma_p = Sigma_p #variance of distance (hidden state)
        self.Sigma_u = Sigma_u #variance of sensory input
        
        self.F = 0 #Free energy!
    
    def g(self, phi, sensor_num):
        #generative model
        if sensor_num == 0:
            return g_true_l1(phi)
        elif sensor_num == 1:
            return g_true_l2(phi)
        elif sensor_num == 2:
            return g_true_l3(phi)
    
    def g_deriv(self, phi, sensor_num):
        if sensor_num == 0:
            return g_true_deriv_l1(phi)
        elif sensor_num == 1:
            return g_true_deriv_l2(phi)
        elif sensor_num == 2:
            return g_true_deriv_l3(phi)


    def calc_free_energy(self, V_p, u):
        """
        Calculates the free energy at the current step
        inputs
        V_p - hierachical prior
        u - sensory input
        """
        #recalculate prediction errors because prediction of hidden state might have been updated
        e_p = (self.phi - V_p)
        e_u = (u - self.g(self.phi, 0))
        F = (e_u**2 / self.Sigma_u + e_u**2 + self.Sigma_p + np.log(self.Sigma_p * self.Sigma_u)) / 2
        return F

    def inference(self, V_p, u):
        """
        #i - timestep
        mu_p - prior guess of hidden state
        u - sensory input
        """

        #page 4 of bogacz for the maths

        #Prediction errors
        e_p = (self.phi - V_p) * (1/self.Sigma_p) #current best guess of distance MINUS prior best guess (epsilon_p in bogacz)
        e_u1 = (u[0] - self.g(self.phi, 0)) * (1/self.Sigma_u[0]) #sensory reading - the light level of the generative model (the difference between sensory input and the inner model) (epsilon_u in bogacz)
        e_u2 = (u[1] - self.g(self.phi, 1)) * (1/self.Sigma_u[1])
        e_u3 = (u[2] - self.g(self.phi, 2)) * (1/self.Sigma_u[2]) #US sensor

        #computation carried out in nodes to compute prediction errors.
        self.eps_p = self.eps_p + dt * (e_p- self.Sigma_p * self.eps_p) # moves towards mean of prior
        self.eps_u1 = self.eps_u1 + dt * (e_u1 - self.Sigma_u[0] * self.eps_u1) # moves according to sensory input
        self.eps_u2 = self.eps_u2 + dt * (e_u2 - self.Sigma_u[1] * self.eps_u2)
        self.eps_u3 = self.eps_u3 + dt * (e_u3 - self.Sigma_u[2] * self.eps_u3)

        self.phi = self.phi + dt * ( 
        - self.eps_p 
        + self.eps_u1 * self.g_deriv(self.phi, 0) 
        + self.eps_u2 * self.g_deriv(self.phi, 1) 
        + self.eps_u3 * self.g_deriv(self.phi, 2) 
        )

        #self.F = self.calc_free_energy(u)

        """
        #calc predictions of sensory input for each sensor
        phi_u = []
        for i in range(0, len(u)):
            phi_u.append(self.g(self.phi, i)) #sensory_predictions, based on feeding the current prediction of the hideen state to the generative model
        """
        phi_u = 0
        return self.phi, phi_u


def run(N, dt, V, V_p, Sigma_p, Sigma_u, sensors):
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
    SENSOR_NUM = len(sensors)#2
    #inititalise arrays for recording results
    phi = np.zeros(N)
    phi_u = np.zeros(N) # prediction of sensory input
    F = np.zeros(N) #free energy
    v = np.zeros(N) # true distance (hidden state)
    u = np.zeros((N, SENSOR_NUM)) #sensory input

    #phi[0] = V_p #in

    for i in range(0, N):
        v[i] = V
        u[i, 0] = sensors[0].ambient_light_intensity #lsensor1
        u[i, 1] = sensors[1].ambient_light_intensity #lsensor2
        u[i, 2] = sensors[2].distance_centimeters #us sensor

        #u[i] = g_true(V) #sensory input given as true generative process generating sensory input
        phi[i], phi_u[i] = robot.inference(V_p, u[i]) #do inference at the current timestep with the previous hierachical prior, and current sensory input
        print('Epoch \r',i+1, '/', N, end="")
    #print("light readings", u)
    return phi, phi_u, v , u

def g_true_l1(x):
    """
    generative function for light sensor 1
    """
    a = s1params_a
    b = s1params_b
    c = s1params_c
    return (a ** (x-b)) + c

def g_true_l2(x):
    """
    generative function for light sensor 3
    """
    a = s2params_a
    b = s2params_b
    c = s2params_c
    return (a ** (x-b)) + c

def g_true_l3(x):
    """
    generative function for light sensor 3
    """
    a = s3params_a
    b = s3params_b
    return (a*x) + b

def g_true_deriv_l1(x):
    a = s1params_a
    b = s1params_b
    return (x-b) * (a**(x-b-1))

def g_true_deriv_l2(x):
    a = s2params_a
    b = s2params_b
    return (x-b) * (a**(x-b-1))

def g_true_deriv_l3(x):
    a = s3params_a
    return a

"""
def write_results_to_csv(filename, phi, phi_u1, phi_u2 ,v ,u ):
    data = {'phi':phi, 'phi_u1':phi_u1, 'phi_u2':phi_u2, 'v':v, 'u':u}    
    df = pd.DataFrame(data)
    df.to_csv(filename)
"""

def write_results_to_json(filename, phi, phi_u ,v ,u):
    """
    Using json write the dictionary to a json file
    """
    data = {'phi':phi, 'phi_u':phi_u, 'v':v, 'u':u} 
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def read_data_from_file_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


"""
calc mean variance from the recordings
"""
data = read_data_from_file_json('data_out.json')

s1_variance = sum(data['s1']['variances'])/len(data['s1']['variances'])
s2_variance = sum(data['s2']['variances'])/len(data['s2']['variances'])
s3_variance = sum(data['s3']['variances'])/len(data['s3']['variances'])

print("variances s1:", s1_variance, "s2:", s2_variance, "s3:", s3_variance)

"""
read params from file
"""
params = read_data_from_file_json('genmap_params.json')

#assign params to globals
s1params_a, s1params_b, s1params_c = params['s1']['a'], params['s1']['b'], params['s1']['c']
s2params_a, s2params_b, s2params_c = params['s2']['a'], params['s2']['b'], params['s2']['c']
s3params_a, s3params_b = params['s3']['a'], params['s3']['b']

"""
initialise sensors
"""
l_sensor1 = ColorSensor(INPUT_1)
l_sensor2 = ColorSensor(INPUT_3)
us_sensor = UltrasonicSensor()

#hyperparams
N = 10000
dt = 0.0001
V = 60 #true hidden state (dist)
V_p = 0 #prior

"""
run the robots inference program
"""
phi, phi_u, v, u = run(N, dt, V, V_p, 1, [s1_variance, s2_variance, s3_variance], [l_sensor1, l_sensor2, us_sensor])
print("final predic", phi[len(phi)-1])

"""
write out the results to file
"""
dataa = {'phi':phi.tolist()} 
with open('phi3sensor.json', 'w') as outfile:
    json.dump(dataa, outfile)
#write_results_to_csv('inference2.csv', phi, phi_u[0], phi_u[1],v, u) #write results out to csv file
print(phi, phi_u)