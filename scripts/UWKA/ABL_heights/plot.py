# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:05:24 2020

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
%matplotlib qt5
#%matplotlib inline
#%% plot parameters

import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
mpl.rcParams['axes.linewidth'] = 2.0 #set the value globally
#sns.set()

plt.rc('font', family='serif') #set font and font sizes
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
#plt.minorticks_on()

plt.rc('text', usetex=False) #set latex rendering True/False


#%%

#Reading in the files

os.chdir(r"C:\Users\Sreenath\Documents\CHEESEHEAD\LiDAR Data\UWKA\20190711a")
dirc = r"C:\Users\Sreenath\Documents\CHEESEHEAD\LiDAR Data\UWKA\20190711a"

list1 = os.listdir(dirc)
print(list1)

#for i in 0:len(list1)
#%% Set Variables used in the script
filename = 'aircraft.UWyo_King_Air.20190711150001.WCLUP_Backscatter_Depol_L1.0.nc'

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
#data.data_vars['CopolPowerR2'].status
#%% pull out power and range data

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

#%% raw data rough look

#FutureWarning: Using an implicitly registered datetime converter for a 
#matplotlib plotting method. The converter was registered by pandas on import.
#Future versions of pandas will require you to explicitly register matplotlib converters.
#To register the converters:

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


fig, ax = plt.subplots(figsize=(15, 4))
plt.pcolormesh(Time2D, Yalt2D, dbPow.T, vmin=-50, vmax=-20)
ax.set_ylabel("Altitude AGL (m)")
cb1 = plt.colorbar(extend='both')
cb1.set_label('Raw Power (dB)')
plt.ylim((1500,5500))

#%% depolarization ratio
dep = np.transpose(np.asarray(data.data_vars['Depolarization']))
fig, ax1 = plt.subplots(figsize=(15, 4))
plt.pcolormesh(Time2D, Yalt2D, dep, vmin=0., vmax=0.5)
plt.ylim((500,5500))
ax.set_ylabel("Altitude AGL (m)")
ax.set_xlabel("UTC Time")
cb1 = plt.colorbar(extend='both')
cb1.set_label('Depolarization (Uncalibrated)')

#%% Process data : Use a window smoothing function from bottleneck (bn).
# Raw power data as a function of the square of the range
# Apply a moving mean smoothing function with widths of 11 time samples and 33 gates
#p1r2 = pd.rolling_mean(p1.mul(r.values**2, axis='columns'), 11)
#p2r2 = pd.rolling_mean(p2.mul(r.values**2, axis='columns'), 11)
p1r2 = p1.mul(r.values**2, axis='columns')
p2r2 = p2.mul(r.values**2, axis='columns')

p1r2_sm = bn.move_mean(p1r2, window=33, axis=1)
p2r2_sm = bn.move_mean(p2r2, window=33, axis=1)
p1r2_sm = bn.move_mean(p1r2_sm, window=11, axis=0)
p2r2_sm = bn.move_mean(p2r2_sm, window=11, axis=0)

# Calculate the depolarization ratio 
#dep   = p2r2_sm/p1r2_sm/depconstA/depconstB
dep = np.transpose(np.asarray(data.data_vars['Depolarization']))

# Create a decibel representation of processed power
dbp1r2 = 10. * ufuncs.log10(p1r2.values)
dbp1r2_sm = 10. * ufuncs.log10(p1r2_sm)

#dbp1r2 = 10. * ufuncs.log10(p1r2.values)
#dbp1r2_sm = 10. * ufuncs.log10(p1r2_sm)

#p1.shape, p1r2.shape, dep.shape, p1r2_sm.shape

#%% Depolarization ratio, uncalibrated

fig, ax1 = plt.subplots(figsize=(45, 12))

plt.subplot(3, 1, 1)
plt.pcolormesh(Time2D, Yalt2D, dep, vmin=0., vmax=0.5)
plt.ylim((100,5000))
ax.set_ylabel("Altitude AGL (m)")
ax.set_xlabel("UTC Time")
cb1 = plt.colorbar(extend='both')
cb1.set_label('Depolarization')

##%% Processed power at the lower levels, no smoothing and then smoothing applied

plt.subplot(3, 1, 2)
#fig, ax = plt.subplots(figsize=(15, 4))
plt.pcolormesh(Time2D, Yalt2D, dbp1r2, vmin=-35, vmax=-10)
plt.ylim((100,5000))
ax.set_ylabel("Altitude AGL (m)")
cb1 = plt.colorbar(extend='both')
cb1.set_label('Power, no smoothing applied (dB)')
#plt.show()
##%%
plt.subplot(3, 1, 3)
#fig, ax = plt.subplots(figsize=(15, 4))
plt.pcolormesh(Time2D, Yalt2D, dbp1r2_sm, vmin=-35, vmax=-10)
plt.ylim((100,5000))
ax.set_ylabel("Altitude AGL (m)")
cb1 = plt.colorbar(extend='both')
cb1.set_label('Power, smoothing applied (dB)')
#plt.show()

#%%

#subset data for reasonable vertical limits


##%% using splien interpolation, initial trial, don't run
##from scipy.interpolate import UnivariateSpline
#
#p1r2_spl = UnivariateSpline(Yalt2D[1199,],np.asarray(p1r2_sm)[1199,],s=0,k=3)
#
##plt.plot(np.asarray(p1r2)[0,],Yalt2D[0,],'r.')
#
#
##plt.plot(x,y,'ro',label = 'data')
#y_range = np.linspace(Yalt2D[0,][0],Yalt2D[0,][-1],1000)
#x_range = np.asarray(p1r2_spl(y_range))
#
#plt.plot(x_range,y_range)
#plt.ylim((1000,3000))
#plt.xlim((-0.05,0.05))
#
##plt.plot(y_range,x_range)
##plt.xlim((1000,3000))
##plt.ylim((-0.05,0.05))
#
#
#dy=np.diff(y_range,1)
#dx=np.diff(x_range,1)
#yfirst=dy/dx
#
#plt.plot(yfirst,y_range[:-1])
#plt.ylim((1000,3000))
#
#y_range[np.argmax(yfirst)]

#x = x_range
#y = np.asarray(p1r2_spl(x_range))
#
#dy=np.diff(y,1)
#dx=np.diff(x,1)
#yfirst=dy/dx
#
#plt.plot(yfirst)

#p1r2_spl_2d = p1r2_spl.derivative(n=2)
#
#plt.plot(p1r2_spl_2d(x_range),x_range)
#plt.ylim((1000,3000))
#plt.xlim((-0.0005,0.0005))
#%% initial trial
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
    
#plt.plot(height)
#plt.ylim((1000,3000))
#plt.xlim((-0.05,0.05))
    
#%%

    
    
#%% checking derived heights with the actual signal
fig, ax = plt.subplots(figsize=(15, 4), sharex=True)
timedata = data.data_vars['Time'].values #array for time 

plt.subplot(2,1,1)
plt.plot(timedata[10:],height)
plt.ylim((1000,4000))

plt.subplot(2,1,2)
plt.pcolormesh(Time2D, Yalt2D, dbp1r2_sm, vmin=-35, vmax=-10)
plt.ylim((1000,3000))
ax.set_ylabel("Altitude AGL (m)")
cb1 = plt.colorbar(extend='both')
cb1.set_label('Power, smoothing applied (dB)')
    
##%% loop check
#check = np.asarray(test['p1r2_sm'])
#
##plt.semilogx(x_range[:-1],y_range[:-1])
#i = 10
#y_range = (Yalt2D[i,][0:])
##x_range = np.asarray(p1r2_spl(y_range))
#x_range = np.asarray(p1r2)[i,]
#plt.plot(x_range,y_range)
#plt.ylim((1000,3000))
#plt.xlim((-0.05,0.05))
#
#data_power = pd.DataFrame()
#data_power['Yalt'] = y_range
#data_power['p1r2_sm'] = x_range
#
#test = pd.DataFrame()
#
#test = data_power[ data_power['p1r2_sm'].notnull()][data_power['Yalt'] < 2500]
#
#plt.plot(test['p1r2_sm'],test['Yalt'])
#plt.ylim((1100,2500))
#plt.xlim((-0.05,0.05))
#
#print(y_range[np.argmax(np.asarray(test['p1r2_sm']))])


#%% check profile data
#fig, ax1 = plt.subplots(figsize=(45, 12))
#plt.plot(len(dbp1r2[0,]))
#plt.subplot(1,2, 1)
#plt.plot(np.asarray(p1r2_sm)[10,],Yalt2D[10,],'r')

a = np.asarray(timedata)[290]
a

plt.plot(np.asarray(p1r2)[1199,],Yalt2D[1199,],'b')

plt.ylim((1070,1075))
plt.xlim((-0.05,0.05))
#plt.title('early morning')
#plt.subplot(1,2, 2)
#plt.plot(np.asarray(p1r2)[1190,],Yalt2D[1190,])
#plt.ylim((1000,3000))
#plt.xlim((-0.05,0.05))
#plt.title('late morning')
#%% first trial with spline interpolation, discarded because too complicated


#from scipy import interpolate
#plt.plot(np.asarray(p1r2)[0,],Yalt2D[0,],'r.')
#plt.ylim((1000,3000))
#plt.xlim((-0.05,0.05))
#
#x_points = np.asarray(p1r2)[0,]
#y_points = Yalt2D[0,]
#
#tck = interpolate.splrep(y_points, x_points)
#
##interpolate.splev(y_points, tck)
#test = interpolate.splev(y_points, tck)
#
#test2 = test[test > 1100 ]| test < 5000]
#
#plt.plot(interpolate.splev(y_points, tck),Yalt2D[0,],'r.')
#plt.ylim((1000,3000))
#plt.xlim((-0.05,0.05))


#%% final plots for signals, run if 2d signal plots are desired

#Processed power at the lower levels, no smoothing 
fig, ax = plt.subplots(figsize=(15, 4))
plt.pcolormesh(Time2D, Yalt2D, dbp1r2, vmin=-50, vmax=-10)
plt.ylim((1500,5000))
ax.set_ylabel("Altitude AGL (m)")
cb1 = plt.colorbar(extend='both')
cb1.set_label('Power, no smoothing applied (dB)')
#%% and then smoothing applied
fig, ax = plt.subplots(figsize=(15, 4))
plt.pcolormesh(Time2D, Yalt2D, dbp1r2_sm, vmin=-50, vmax=-10)
plt.ylim((1000,5000))
ax.set_ylabel("Altitude AGL (m)")
cb1 = plt.colorbar(extend='both')
cb1.set_label('Power, smoothing applied (dB)')

