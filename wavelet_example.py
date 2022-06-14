import Waveletfunctions as wf
from matplotlib.gridspec import GridSpec
from scipy.special._ufuncs import gammainc, gamma
import numpy as np
from scipy.optimize import fminbound

import kappa_fitting as kf

import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter


dt = 24.0
tvals = np.arange(200)*dt

P1 =  400
P2 =  100

test_input = np.sin((2*np.pi/P1)*tvals) + 2e-4*tvals*np.sin((2*np.pi/P2)*tvals)

v = kf.timeseries(test_input,dt=dt)

scaleavg=[80,120]

sst, power, period, global_ws, scale_avg, sig95, coi, global_signif = wf.waveletanalysis(v.data,v.dt,plot=False,scaleavg=scaleavg)

time = np.arange(0,v.n)*v.dt
xlim = ([time[0], time[-1]])
#--- Plot time series
fig = plt.figure(figsize=(10, 10))
fig.canvas.set_window_title('Wavelet plots')
gs = GridSpec(3, 4, hspace=0.4, wspace=0.9)
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0, hspace=0)
plt1 = plt.subplot(gs[0, 0:3])
plt.plot(time, sst, 'k')
plt.xlim(xlim[:])
plt.xlabel('Time [s]')
plt.ylabel('Intensity [arb.]')
plt.title('a) Timeseries plot')

#--- Contour plot wavelet power spectrum

plt3 = plt.subplot(gs[1, 0:3],sharex=plt1)


# - determine colormap levels
#--------------------------------------------------
order = round(np.log10(np.max(power))) - 1
lvlsteps = 10**order
mx = np.ceil(np.max(power) * 10**-order + 1)*10**order
no_lvls = mx/lvlsteps
while no_lvls > 10:
    lvlsteps *= 2
    no_lvls = mx/lvlsteps


levels = np.arange(0,mx,lvlsteps)
#--------------------------------------------------

#im = plt.contourf(time, period, power, len(levels), levels=levels,cmap="magma")        
im = plt.pcolormesh(time, period, np.log10(power),cmap='magma')
plt.xlabel('Time [s]')
plt.ylabel('Period [s]')
plt.title('b) Wavelet Power Spectrum.')
plt.xlim(xlim[:])

# 95# significance contour, levels at -99 (fake) and 1 (95# signif)
plt.contour(time, period, sig95, [-99, 1], colors='blue')

# cone-of-influence, anything "below" is dubious
plt.plot(time, coi[:], 'yellow')

# format y-scale
plt3.set_yscale('log', basey=2, subsy=None) # set scale
plt.ylim([np.min(period), np.max(period)])  # set limits
ax = plt.gca().yaxis                         
ax.set_major_formatter(ticker.ScalarFormatter())
plt3.ticklabel_format(axis='y', style='plain') # ticks
plt3.invert_yaxis()

# set up the size and location of the colorbar
position=fig.add_axes([0.675,0.40,0.01,0.2])     # can adjust this for different position and size
cb = plt.colorbar(im, cax=position, orientation='vertical', fraction=0.01, pad=0.5)
cb.ax.tick_params(labelsize=8) # change tick label fontsize


#--- Plot global wavelet spectrum
plt4 = plt.subplot(gs[1, -1],sharey=plt3)
plt.plot(global_ws, period)
plt.plot(global_signif, period, '--')
plt.xlabel('Power')
plt.title('c) Global Wavelet Spectrum')
plt.xlim([0, 1.25 * np.max(global_ws)])
# format y-scale
plt4.set_yscale('log', basey=2, subsy=None)
plt.ylim([np.min(period), np.max(period)])
ax = plt.gca().yaxis
ax.set_major_formatter(ticker.ScalarFormatter())
plt4.ticklabel_format(axis='y', style='plain')
plt4.invert_yaxis()

# --- Plot 2--8 yr scale-average time series
plt.subplot(gs[2, 0:3],sharex=plt1)
plt.plot(time, scale_avg, 'k')
plt.xlim(xlim[:])
plt.xlabel('Time [s]')
plt.ylabel('Avg variance')
plt.title('d) %.0d-%.0d second Scale-average Time Series'%(scaleavg[0],scaleavg[1]))
#plt.plot(xlim, scaleavg_signif + [0, 0], '--')

plt.show()
