# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:12:27 2020

@author: Sreenath
"""

import xarray
from xarray import ufuncs
import pandas as pd
import numpy as np
import bottleneck as bn
import os
import matplotlib 
import matplotlib.pyplot as plt
#%matplotlib qt5
%matplotlib inline
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#iop01
dirwrk = "D:\Sreenath\CHEESEHEAD\UWKA\Cloud LiDAR\20190711b" 
os.chdir(rdirwrk)
dirc = rdirwrk

list1 = os.listdir(dirc)
print(list1)
#%% 
for i in range(len(list1)):

    filename = list1[i]

# Constants applied to get Depolarization factor, different for every case
# Best guess here
#depconstA = 3.67589
#depconstB = 4.5

# OPen .nc file

    data=xarray.open_dataset(filename, decode_times=True)
#data.dims
#data.coords
    data.data_vars['CopolPowerR2'].status
    data.Range

##%% pull out power and range data

    r = xarray.DataArray((data.data_vars['Range']/1000.)).to_dataframe().squeeze()

    p1 = pd.DataFrame(data=np.asarray(data.data_vars['CopolPowerR2'].values.T), index=np.asarray(data.data_vars['Time']),
                                   columns=np.asarray(data.data_vars['Range']))
    p2 = pd.DataFrame(data=np.asarray(data.data_vars['CrossPowerR2'].values.T), index=np.asarray(data.data_vars['Time']),
                                   columns=np.asarray(data.data_vars['Range']))

# Now let's make 2D arrays for plotting
    Yalt2D, Alt2D = np.meshgrid(data.data_vars['Range'].values, data.data_vars['ALT'].values)
    Time2D = np.empty_like(Yalt2D, dtype=data.data_vars['Time'].dtype)
# This next step could use some speeding up, but the datatype doesn't allow np.meshgrid call
    for i in range(Yalt2D.shape[1]):
        Time2D[:,i] = data.data_vars['Time'].values
        data.data_vars['Time'].shape, Yalt2D.shape
#junk, Time2Db= np.meshgrid(data.data_vars['Range'].values, np.array(data.data_vars['Time'].values, dtype=np.datetime64))


# Create an array with the common expression of lidar power
# Note: Do not use this going forward with data manipulation, as we are making this non-linear
# Only for display purposes
    dbPow = 10. * ufuncs.log10(data.data_vars['CopolPowerR2'].values)

    Yalt2D = Yalt2D + Alt2D

##%% Process data : Use a window smoothing function from bottleneck (bn).
# Raw power data as a function of the square of the range
# Apply a moving mean smoothing function with widths of 11 time samples and 33 gates
    p1r2 = p1.mul(r.values**2, axis='columns')
    p2r2 = p2.mul(r.values**2, axis='columns')

    p1r2_sm = bn.move_mean(p1r2, window=33, axis=1)
    p2r2_sm = bn.move_mean(p2r2, window=33, axis=1)
    p1r2_sm = bn.move_mean(p1r2_sm, window=11, axis=0)
    p2r2_sm = bn.move_mean(p2r2_sm, window=11, axis=0)

# Calculate the depolarization ratio 
    dep = np.transpose(np.asarray(data.data_vars['Depolarization']))

# Create a decibel representation of processed power
    dbp1r2 = 10. * ufuncs.log10(p1r2.values)
    dbp1r2_sm = 10. * ufuncs.log10(p1r2_sm)

    height = []
    for i in range(10,(len(p1r2_sm))):
    #spline interpolation
    #p1r2_spl = UnivariateSpline(Yalt2D[i,],np.asarray(p1r2_sm)[i,],s=0,k=3) 
    
    
        y_range = (Yalt2D[i,][0:])
    #x_range = np.asarray(p1r2_spl(y_range))
    #x_range = np.asarray(p1r2_sm)[i,]
        x_range = np.asarray(p1r2)[i,]
    
        data_power = pd.DataFrame()
        data_power['Yalt'] = y_range
    #data_power['p1r2_sm'] = x_range
        data_power['p1r2'] = x_range
    
        test = pd.DataFrame()
    #check height range for Yalt
    #test = data_power[ data_power['p1r2_sm'].notnull()][data_power['Yalt'] < 2600]
        test = data_power[ data_power['p1r2'].notnull()][data_power['Yalt'] < 2600]
    
    
    
    #h = (test[test['p1r2_sm'] == max(test['p1r2_sm'])]['Yalt'])
        h = (test[test['p1r2'] == max(test['p1r2'])]['Yalt'])
    
        height = np.append(height,h)
    
        print(h)
    
    
##%% checking derived heights with the actual signal
    fig, ax = plt.subplots(figsize=(16, 6), sharex=True)
    timedata = data.data_vars['Time'].values[10:] #array for time 

    plt.subplot(2,1,1)
    plt.plot(timedata,height)
    plt.ylim((1000,4000))

    plt.subplot(2,1,2)
    plt.pcolormesh(Time2D, Yalt2D, dbp1r2_sm, vmin=-35, vmax=-10)
    plt.ylim((1000,3000))
    ax.set_ylabel("Altitude AGL (m)")
    cb1 = plt.colorbar(extend='both')
    cb1.set_label('Power, smoothing applied (dB)')
    plt.savefig((filename + '.png'),dpi=600);

    abl_height = pd.DataFrame()
    abl_height['time'] = timedata
    abl_height['height'] = height
    abl_height.to_csv((filename + '.csv'))

#%%end loop    
####