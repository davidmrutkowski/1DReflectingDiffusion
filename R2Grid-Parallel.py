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
import sys

# number of steps to approximate infinite sum
max_n = 500
    
x_d = 0.0
x_e = 0.0


print(sys.argv)

I0 = float(sys.argv[2])

I1 = I0
Iinf = float(sys.argv[3])

filepath = sys.argv[1]

#1RitC imaging time, see Feb 21, 2020 email from Veneta
t_step_size = 0.729

x_initial = []
z_initial = []

external_dist = 0.0

include_neighbor_count = 0

max_num_points = 30
    
def diffusion_reflect(data, D, koff):
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


    
#https://stackoverflow.com/questions/55030369/python-surface-fitting-of-variables-of-different-dimensionto-get-unknown-paramet
graphWidth = 800 # units are pixels
graphHeight = 600 # units are pixels

# 3D contour plot lines
numberOfContourLines = 16

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
        os.mkdir("Images")
    except OSError as error:
        print(error)
    
    for key,value in z_by_time_dict.items():
        plt.plot(x_initial, z_by_time_dict[key], color='b', marker='.')
        plt.plot(unique_x_data_extended, z_predicted_by_time_dict[key], color='r', linestyle='--')

        plt.title("t = " + str(key))
        plt.ylim([0.0,150.0])
        plt.xlabel(r'Position [$\mu$m]')
        plt.ylabel('Intensity [AU]')
        plt.savefig('Images\\' + str(key) + '.png')
        plt.clf()
        # plt.show()
        
    
    
def SurfacePlot(func, data, fittedParameters):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)

    axes = Axes3D(f)

    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = np.linspace(min(x_data), max(x_data), 100)
    yModel = np.linspace(min(y_data), max(y_data), 100)
    X, Y = np.meshgrid(xModel, yModel)

    Z = func(np.array([X, Y]), *fittedParameters)

    surface = axes.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='grey', alpha=0.9, linewidth=1, zorder=5, antialiased=True)
    
    above_prediction = [[],[],[]]
    below_prediction = [[],[],[]]
    error_list = []
    error_list_above = []
    error_list_below = []
    z_predicted_list = []
    
    z_predicted_list = func(np.array([x_data, y_data]), *fittedParameters)
    for i in range(0, len(z_predicted_list)):        
        error_list.append(z_data[i] - z_predicted_list[i])
        
        if z_data[i] <= z_predicted_list[i]:
            below_prediction[0].append(x_data[i])
            below_prediction[1].append(y_data[i])
            below_prediction[2].append(z_data[i])
            error_list_below.append(z_data[i] - z_predicted_list[i])
        else:
            above_prediction[0].append(x_data[i])
            above_prediction[1].append(y_data[i])
            above_prediction[2].append(z_data[i])
            error_list_above.append(z_data[i] - z_predicted_list[i])
    
    max_cbar = 10.0
    if len(error_list_below) <= 0:
        max_cbar = abs(max(error_list_above))
    elif len(error_list_above) <= 0:
        max_cbar = abs(min(error_list_below))
    else:
        max_cbar = max(abs(min(error_list_below)), abs(max(error_list_above)))

    #points = axes.scatter(x_data, y_data, z_predicted_list, c=error_list, cmap=cm.seismic, vmin=-max_cbar, vmax=max_cbar) # show data along with plotted surface
    points = axes.scatter(above_prediction[0], above_prediction[1], above_prediction[2], s=5.0, c=error_list_above, cmap=cm.seismic, vmin=-max_cbar, vmax=max_cbar, zorder=10, edgecolors='black') # show data along with plotted surface
    points_2 = axes.scatter(below_prediction[0], below_prediction[1], below_prediction[2], s=5.0, c=error_list_below, cmap=cm.seismic, vmin=-max_cbar, vmax=max_cbar, zorder=0, edgecolors='black') # show data along with plotted surface
    cbar = f.colorbar(points, shrink=0.5, ax=axes)

    axes.set_xlabel('position [um]') # X axis data label
    axes.set_ylabel('time [s]') # Y axis data label
    axes.set_zlabel('Intensity [AU]') # Z axis data label
    
    #print(cbar.get_clim())
    plt.show()
    plt.close('all') # clean up after using pyplot or else thaere can be memory and process problems

"""root = tk.Tk()
root.withdraw()

filepath = filedialog.askopenfilename()
root.destroy()"""

x = []
z = []
t = []

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
            
        line = fp.readline()
     
x_d = x_initial[0]
x_e = x_initial[-1]

#external_dist = 5.408

x = np.array(x)
z = np.array(z)
t = np.array(t)

t = t * t_step_size


min_x_l = 0.0
max_x_l = 500.0


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

Iinf_min = (result + min_x_l*I0 + min_x_l*I1) / (2*min_x_l + x_e - x_d)
Iinf_max = (result + max_x_l*I0 + max_x_l*I1) / (2*max_x_l + x_e - x_d)


D_guess = 0.001
koff_guess = 0.0033
Iinf_guess = 0.5*(Iinf_min+Iinf_max)

initialParams = [D_guess, koff_guess, Iinf_guess]

data = [x, t, z]

D_list = np.logspace(np.log10(1e-6), np.log10(1.0), 120)
koff_list = np.logspace(np.log10(1e-6), np.log10(1.0), 120)

output_file = open(filepath + "_R2-Grid-D-koff_I0-" + str(I0) + "_Iinf-" + str(Iinf) + "_tstep-" + str(t_step_size) + "-frames-" + str(max_num_points) + "_xlLimited.csv", "w")
output_file.write("D,koff,R^2\n")

for i in koff_list:
    print(np.where(koff_list==i), len(koff_list))
    curr_koff = i
    
    temp_array = []
    for j in D_list:
        curr_D = j
        
        modelPredictions = diffusion_reflect(data[:2], curr_D, curr_koff)

        residuals = modelPredictions - z

        SS_res = np.sum(residuals**2) # squared errors
        SS_tot = np.sum((z-np.mean(z))**2)

        Rsquared = 1.0 - (SS_res / SS_tot)
        
        output_file.write("{0},{1},{2}\n".format(curr_D, curr_koff, Rsquared))
        temp_array.append(Rsquared)