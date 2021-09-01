"""
 * Copyright (C) 2021 Lehigh University.
 *
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see http://www.gnu.org/licenses/.
 *
 * Author: David Rutkowski (dmr518@lehigh.edu)
 *
 * Details of model used is described in further detail in Gerganova et al.
 * https://www.biorxiv.org/content/10.1101/2020.12.18.423457v3
"""
 

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import math
import scipy, scipy.optimize
import matplotlib
from mpl_toolkits.mplot3d import  Axes3D
from matplotlib import cm
import random
import os
import tkinter as tk
from tkinter import filedialog

# number of steps to approximate infinite sum
max_n = 500

# boundary positions of data, will be determined from input file    
x_d = 0.0
x_e = 0.0

#assumed flat intensity value beyond observed data
#can be estimated by average line intensity before bleach
I0 = 718.4446379
I1 = I0
#Iinf = 41.18656861 

#step size between timesteps of frames
t_step_size = 10.0

#cutoff to limit analysis to only this number of timesteps
max_num_points = 1000

#initial guess of D and koff, does not need to be very accurate
D_guess = 0.000810514
koff_guess = 0.000872898

#minimum and maximum values for length of side regions, to restrict solver to realistic sizes
min_l_f = 2.0
max_l_f = 10.0

x_initial = []
z_initial = []

#number of additional neighbors to include in averaging
include_neighbor_count = 0

cell_name = ""
    
def diffusion_reflect(data, D, koff, Iinf):
    x = data[0]
    t = data[1]
    
    result = 0.0
    for i in range(0, len(x_initial)-1):
        avg_intensity = 0.0
        avg_intensity_count = 0
        for j in range(i-include_neighbor_count, i+include_neighbor_count+2):
            if j >= 0 and j < len(x_initial):
                avg_intensity += z_initial[j]
                avg_intensity_count += 1
                
        avg_intensity = avg_intensity / float(avg_intensity_count)
            
        x_diff = x_initial[i+1]-x_initial[i]
        
        result += avg_intensity*x_diff
    
    x_l = (Iinf*(x_d-x_e) + result) / (2*Iinf-I0-I1)
    #x_l = 25
    
    print("D = {0:0.6f}, koff = {1:0.6f}, Iinf = {2:0.6f}, x_l = {3:0.6f}".format(D, koff, Iinf, x_l))
    
    if x_l < 0.0:
        print("x_l less than zero: " + str(x_l) + ", Iinf may not be maintained")
        exit()
        x_l = 0.0
        
        x_a = x_d
        x_b = x_e
    else:
        x_a = x_d - x_l
        x_b = x_e + x_l
    
        
     
    x_diff = x_d - x_a
    result += I0 * x_diff
    
    x_diff = x_b - x_e
    result += I1 * x_diff
    
    result = result / (x_b-x_a) 
    
    sums = [0.0] * (len(x_initial)-1)
    sum_left = 0.0
    sum_right = 0.0
    
    result_two = 0.0
    
    for n in range(1, max_n+1):
        lambda_n = n*math.pi / (x_b -x_a)
        middle_term = np.exp(-D*t*lambda_n*lambda_n) / lambda_n * np.cos(lambda_n * (x-x_a))
        
        for i in range(0, len(x_initial)-1):
            avg_intensity = 0.0
            avg_intensity_count = 0
            for j in range(i-include_neighbor_count, i+include_neighbor_count+2):
                if j >= 0 and j < len(x_initial):
                    avg_intensity += z_initial[j]
                    avg_intensity_count += 1
                    
            avg_intensity = avg_intensity / float(avg_intensity_count)
            
            sums[i] += avg_intensity * middle_term * (np.sin(lambda_n*(x_initial[i+1] - x_a)) - np.sin(lambda_n*(x_initial[i] - x_a)))
    
        sum_left += I0 * middle_term * (np.sin(lambda_n*(x_d - x_a)) - np.sin(lambda_n*(x_a - x_a)))
        sum_right += I1 * middle_term * (np.sin(lambda_n*(x_b - x_a)) - np.sin(lambda_n*(x_e - x_a)))
        
    for i in range(0, len(x_initial)-1):
        result_two += sums[i] 
    
    result_two += sum_left
    result_two += sum_right
    
    result_two = 2.0 * np.exp(-koff*t) / (x_b - x_a)  * result_two
    
    return result + result_two


def IndividualLineComparisons(func, data, fittedParameters):
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]
    
    x_step = 0.042491803

    x_a = x_d
    x_b = x_e
    
    result = 0.0
    for i in range(0, len(x_initial)-1):
        avg_intensity = 0.0
        avg_intensity_count = 0
        for j in range(i-include_neighbor_count, i+include_neighbor_count+2):
            if j >= 0 and j < len(x_initial):
                avg_intensity += z_initial[j]
                avg_intensity_count += 1
                
        avg_intensity = avg_intensity / float(avg_intensity_count)
            
        x_diff = x_initial[i+1]-x_initial[i]
        
        result += avg_intensity*x_diff
    
    Iinf = fittedParameters[2]
    
    x_l = (Iinf*(x_d-x_e) + result) / (2*Iinf-I0-I1)

    if x_l < 0.0:
        print("x_l less than zero: " + str(x_l) + ", Iinf may not be maintained")
        x_l = 0.0
        
        x_a = (I0*x_d  - Iinf*x_e + result) / (I0 - Iinf)
        x_b = x_e
    else:
        x_a = x_d - x_l
        x_b = x_e + x_l
        
    lower_range = np.arange(x_a, x_d, x_step).tolist()
    upper_range = np.arange(x_e+x_step, x_b, x_step).tolist()
    
    x_data_extended = x_data[:].tolist()
    y_data_extended = y_data[:].tolist()

    unique_times = list(set(y_data))
    
    temp_list_x = []
    temp_list_y = []
    for i in range(0, len(lower_range)):
        for j in range(0, len(unique_times)):
            temp_list_x.append(lower_range[i])
            temp_list_y.append(unique_times[j])
            
    temp_list_x.extend(x_data_extended)
    temp_list_y.extend(y_data_extended)
    
    x_data_extended = temp_list_x[:]
    y_data_extended = temp_list_y[:]
    
    temp_list_x = []
    temp_list_y = []
    for i in range(0, len(upper_range)):
        for j in range(0, len(unique_times)):
            temp_list_x.append(upper_range[i])
            temp_list_y.append(unique_times[j])
            
    x_data_extended.extend(temp_list_x)
    y_data_extended.extend(temp_list_y)
    
    z_predicted_list = func(np.array([x_data_extended, y_data_extended]), *fittedParameters)
    
    
    z_by_time_dict = {}
    z_predicted_by_time_dict = {}
    z_predicted_by_time_dict_low = {}
    z_predicted_by_time_dict_high ={}
    
    for i in range(0, len(x_data)):
        curr_time = y_data[i]
        
        if curr_time in z_by_time_dict:
            z_by_time_dict[curr_time].append(z_data[i])
        else:
            z_by_time_dict[curr_time] = [z_data[i]]
    
    for i in range(0, len(x_data_extended)):
        curr_time = y_data_extended[i]
        
        if curr_time in z_predicted_by_time_dict:
            z_predicted_by_time_dict[curr_time].append(z_predicted_list[i])
        else:
            z_predicted_by_time_dict[curr_time] = [z_predicted_list[i]]
    

    unique_x_data_extended = sorted(list(set(x_data_extended)))

    try:
        if not os.path.isdir("Images\\"):
            os.mkdir("Images\\")
        if not os.path.isdir("Images\\"+cell_name):
            os.mkdir("Images\\"+cell_name)
    except OSError as error:
        print(error)
    
    count = 0
    for key,value in z_by_time_dict.items():
        plt.plot(x_initial, z_by_time_dict[key], color='b', marker='.')
        
        if count < max_num_points:
            plt.plot(unique_x_data_extended, z_predicted_by_time_dict[key], color='r', linestyle='--')
        else:
            plt.plot(unique_x_data_extended, z_predicted_by_time_dict[key], color='g', linestyle='--')
        
        plt.title("t = {:.2f} s".format(key))
        plt.ylim([0.0,I0*1.25])
        plt.xlabel(r'Position [$\mu$m]')
        plt.ylabel('Intensity [AU]')
        plt.savefig("Images\\"+cell_name + "\\" + str(key) + '.png')
        #plt.show()
        count+= 1
        

root = tk.Tk()
root.withdraw()

filepath = filedialog.askopenfilename()

cell_name = filepath[filepath.rfind('/')+1:]

pos_TempData = cell_name.find('TempData')
if pos_TempData >= 0:
    cell_name = cell_name[pos_TempData+len('TempData')+1:]

root.destroy()

x = []
z = []
t = []

full_x = []
full_z = []
full_t = []



with open(filepath) as fp:
    line = fp.readline()
    
    while line:
        splitString = line.split()
        
        x_initial.append(float(splitString[0]))
        z_initial.append(float(splitString[1]))
        
        for i in range(1, len(splitString)):
            if i <= max_num_points:
                x.append(float(splitString[0]))
                z.append(float(splitString[i]))
                t.append(i-1)
                
            full_x.append(float(splitString[0]))
            full_z.append(float(splitString[i]))
            full_t.append(i-1)
            
        line = fp.readline()
     
x_d = x_initial[0]
x_e = x_initial[-1]

x = np.array(x)
z = np.array(z)
t = np.array(t)

full_x = np.array(full_x)
full_z = np.array(full_z)
full_t = np.array(full_t)

t = t * t_step_size
full_t = full_t * t_step_size


result = 0.0
for i in range(0, len(x_initial)-1):
    avg_intensity = 0.0
    avg_intensity_count = 0
    for j in range(i-include_neighbor_count, i+include_neighbor_count+2):
        if j >= 0 and j < len(x_initial):
            avg_intensity += z_initial[j]
            avg_intensity_count += 1
            
    avg_intensity = avg_intensity / float(avg_intensity_count)
        
    x_diff = x_initial[i+1]-x_initial[i]
    
    result += avg_intensity*x_diff

Iinf_min = (result + min_l_f*I0 + min_l_f*I1) / (2*min_l_f + x_e - x_d)
Iinf_max = (result + max_l_f*I0 + max_l_f*I1) / (2*max_l_f + x_e - x_d)


Iinf_guess = 0.5*(Iinf_min+Iinf_max)

initialParams = [D_guess, koff_guess, Iinf_guess]

print(Iinf_min, Iinf_max, Iinf_guess)

data = [full_x, full_t, full_z]


fittedParameters, pcov = scipy.optimize.curve_fit(diffusion_reflect, [x, t], z, p0 = initialParams, bounds=([0.0,0.0,Iinf_min], [np.inf,np.inf,Iinf_max]))
#fittedParameters = initialParams

modelPredictions = diffusion_reflect(data[:2], *fittedParameters) 

residuals = modelPredictions - full_z

"""output_temp = open("TEMPFILE.txt", "w")
for i in range(0, len(modelPredictions)):
    output_temp.write('{0} {1} {2}\n'.format(full_t[i], modelPredictions[i], full_z[i]))"""

SS_res = np.sum(residuals**2) # squared errors
SS_tot = np.sum((full_z-np.mean(full_z))**2)

Rsquared = 1.0 - (SS_res / SS_tot)

max_time = int(np.max(full_t)/t_step_size)
Rsquared_array = []

# calc Rsquared for each curve fit and then average
for i in range(0, max_time):
    z_vals = []
    modelPredictions_vals = []
    
    for j in range(0, len(full_t)):
        if full_t[j] == i*t_step_size:
            z_vals.append(full_z[j])
            modelPredictions_vals.append(modelPredictions[j])
    
    if len(z_vals) > 0:
        modelPredictions_vals = np.array(modelPredictions_vals)
        z_vals = np.array(z_vals)
        
        residuals = modelPredictions_vals - z_vals

        SS_res = np.sum(residuals**2) # squared errors
        SS_tot = np.sum((z_vals-np.mean(z_vals))**2)

        Rsquared_array.append(1.0 - (SS_res / SS_tot))
        
#print("R-squared_mean = " + str(np.mean(Rsquared_array)))
#print(filepath)
#print('fitted prameters', fittedParameters)
print("D", fittedParameters[0])
print("koff", fittedParameters[1])
print('R-squared', Rsquared)
print('Iinf', fittedParameters[2])
print('x_l', (fittedParameters[2]*(x_d-x_e) + result) / (2*fittedParameters[2]-I0-I1))
print('I0', I0)

IndividualLineComparisons(diffusion_reflect, data, fittedParameters)