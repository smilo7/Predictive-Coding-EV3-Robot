import json

from ev3dev2.motor import OUTPUT_A, OUTPUT_C, MoveDifferential, SpeedRPM
from ev3dev2.wheel import EV3EducationSetTire

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

def drive_motors(distance):
    movediff = MoveDifferential(OUTPUT_A, OUTPUT_C, EV3EducationSetTire, 10 * 8)
    movediff.on_for_distance(SpeedRPM(20), distance)
