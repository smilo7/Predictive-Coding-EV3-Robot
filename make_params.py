#!/usr/bin/env python3
"""
Read the light recordings data in and calculate the parameters for the generative mapping for each sensor
author: mb743
"""

import json
import numpy as np
from scipy.optimize import curve_fit

def read_data_from_file_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def objective_func(x, a, b, c):
    """
    objective function for to fit the curve for the generative mapping
    """
    return (a ** (x-b)) + c

def objective_func_linear(x, a, b):
    """
    for the US sensor
    """
    return (a*x) + b


def fit_function_to_data(distance, light):
    """
    fits a function to the data and then returns the parameters
    input
    distance - array of distance intervals where recordigs were taken
    light - array of light level recordings
    returns
    parameters a, b and c for curve
    """
    params, _ = curve_fit(objective_func, distance, light)
    a, b, c = params
    print("y = (",a,"(^x-",b,")) +",c)
    x_line = np.arange(min(distance), max(distance))
    y_line = objective_func(x_line, a, b, c)
    return a, b, c

def fit_function_to_data_linear(distance, light):
    """
    fits a function to the data and then returns the parameters
    input THIS IS FOR THE US SENSOR as measurements are linear over distance
    distance - array of distance intervals where recordigs were taken
    light - array of light level recordings
    returns
    parameters a, b and c for curve
    """
    params, _ = curve_fit(objective_func_linear, distance, light)
    a, b = params
    print("y=", a ,"x", "+", b)
    x_line = np.arange(min(distance), max(distance))
    y_line = objective_func_linear(x_line, a, b)
    return a, b



"""
!!!!
FILE TO READ IN
!!!!
"""
data = read_data_from_file_json('data_out.json')

"""
!!!
DISTANCE INTERVALS
CHANGE IF NEEDED
!!!
"""
distances = np.arange(10, 115, 5) #distance intervals
#distances = np.arange(20, 85, 5)

s1_means = data['s1']['means'] #means sensor 1 light sensor 1
s2_means = data['s2']['means'] #light sesnor 2
s3_means = data['s3']['means'] #US sensor

s1_variance = sum(data['s1']['variances'])/len(data['s1']['variances'])
s2_variance = sum(data['s2']['variances'])/len(data['s2']['variances'])
s3_variance = sum(data['s3']['variances'])/len(data['s3']['variances'])


s1params_a, s1params_b, s1params_c = fit_function_to_data(distances, s1_means)
s2params_a, s2params_b, s2params_c = fit_function_to_data(distances, s2_means)
s3params_a, s3params_b = fit_function_to_data_linear(distances, s3_means)


params = {'s1':
        {
        'a':s1params_a, 
        'b':s1params_b, 
        'c':s1params_c
        },
        's2':
        {
        'a':s2params_a, 
        'b':s2params_b, 
        'c':s2params_c
        },
        's3':
        {
        'a':s3params_a, 
        'b':s3params_b
        }
       }

#write the found parameters to file
with open('genmap_params.json', 'w') as outfile:
    json.dump(params, outfile)



########################################
# MAKE LOOKUP TABLE FOR KALMAN FILTER  #
########################################

"""
lookup_range = np.arange(10, 110, 1) # lookup range for light sensor values
lookup_table = {'s1': dict.fromkeys(lookup_range), 's2':dict.fromkeys(lookup_range),'s3': dict.fromkeys(lookup_range)}
for key, value in lookup_table.items():
    for k, v in value.items():
        if key == 's1':
            lookup_table[key][k] = objective_func(k, s1params_a, s1params_b, s1params_c)
        elif key == 's2':
            lookup_table[key][k] = objective_func(k, s2params_a, s2params_b, s2params_c)
        elif key == 's3':
            lookup_table[key][k] = objective_func_linear(k, s3params_a, s3params_b)
"""

def make_lookup():
    dists = np.arange(0, 111, 1) # dist vals

    #get the sensor values for each possible distance in the range
    s1_vals = {int(round(objective_func(d, s1params_a, s1params_b, s1params_c))):d for d in dists}
    s2_vals = {int(round(objective_func(d, s2params_a, s2params_b, s2params_c))):d for d in dists}
    s3_vals = {int(round(objective_func_linear(d, s3params_a, s3params_b))):d for d in dists}

    table = {'s1': s1_vals, 's2':s2_vals,'s3': s3_vals}
    return table

            
lookup_table = make_lookup()
print(lookup_table)
# convert key values to string cos of json
lookup_tablejson = {}
for k, v in lookup_table.items():
    lookup_tablejson[k] = {str(key): str(value) for key, value in v.items()}
print(lookup_tablejson)

#write lookup table as json file out
with open('lookup_table.json', 'w') as outfile:
    json.dump(lookup_tablejson, outfile)

print("done. program finished :)")