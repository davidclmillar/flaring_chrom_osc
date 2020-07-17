from __future__ import print_function, division
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
minutes = mdates.MinuteLocator()
from matplotlib import dates
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
import pandas
from sunpy.timeseries import TimeSeries
from sunpy.time import TimeRange#, parse_time
#from sunpy.net import hek, Fido, attrs as a
import seaborn as sns
palette = sns.set_palette("colorblind")
clrs = sns.color_palette(palette)

# get the goes timeseries from file
tr = TimeRange(['2014-09-06 16:15', '2014-09-06 17:30'])
files = "go1520140906.fits"
goes = TimeSeries(files)
goestrunc = goes.truncate(tr)
times = goestrunc.index
goesdatetimes = [x.to_pydatetime() for x in times]
goes_intensity = goestrunc.data['xrsb'] # isolate one band

# load in integrated crisp data
ca8542times = pickle.load( open( "ca8542times.p", "rb" ) )
ca8542intensity = pickle.load( open( "ca8542intensity.p", "rb" ) )
Halphatimes = pickle.load( open( "Halphatimes.p", "rb" ) )
Halphaintensity = pickle.load( open( "Halphaintensity.p", "rb" ) )

# time axes
plot_datesca8542 = dates.date2num(ca8542times)
plot_datesHalpha = dates.date2num(Halphatimes)
plot_datesgoes   = dates.date2num(goesdatetimes)
myFmt = dates.DateFormatter('%H:%M')

#-------------
# plot
#-------------
f,ax = plt.subplots(figsize=(8,4.5))
ax.plot_date(plot_datesHalpha, Halphaintensity/max(Halphaintensity), linestyle='-',marker=None,color=clrs[4],label="H-alpha integrated")
ax.plot_date(plot_datesca8542, ca8542intensity/max(Halphaintensity), linestyle='-',marker=None,color=clrs[1],label="Ca II 8542 integrated")
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('2014-09-06')
ax.set_ylabel('Intensity (normalised)')
ax.xaxis.set_minor_locator(minutes)
ax.set_ylim(0.8,1.02)
ax.set_yticks([0.8,0.9,1.0])
goescolor=clrs[0]


# add the filled boxes showing pre flare and post impulsive
plt.axvline(ca8542times[0],color=clrs[2],linestyle='--',linewidth=0.7)
plt.axvline(ca8542times[147],color=clrs[2],linestyle='--',linewidth=0.7)

plt.axvline(ca8542times[-147],color=clrs[6],linestyle='--',linewidth=0.7)
plt.axvline(ca8542times[-1],color=clrs[6],linestyle='--',linewidth=0.7)

ax.fill_between([ca8542times[0],ca8542times[147]],0,ax.get_ylim()[1],color=clrs[2],alpha=0.05)
ax.fill_between([ca8542times[-147],ca8542times[-1]],0,ax.get_ylim()[1],color=clrs[6],alpha=0.05)

# other y axis for goes scale
axtwin = ax.twinx()
axtwin.plot_date(plot_datesgoes,goes_intensity, color=goescolor,linestyle='-',marker=None)#,label=r"GOES 1.0 "+u"\u2212"+" 8.0 Å")
axtwin.xaxis.set_major_formatter(myFmt)
axtwin.xaxis.set_minor_locator(minutes)
axtwin.tick_params(axis='y',which='both',color=goescolor,labelcolor=goescolor)
axtwin.set_yscale('log')
axtwin.set_ylim(10**-6.1,10**-3.8)
axtwin.spines['right'].set_color(goescolor)
l = axtwin.set_ylabel(r'W m$^{-2}$',labelpad=15,color=goescolor)
l.set_rotation(270)

# legend
handles, labels = ax.get_legend_handles_labels()
#goeshandle = axtwin.get_legend_handles_labels()
line = Line2D([0], [0], color=clrs[0], linewidth=1.5, linestyle='-',label=r"GOES 1.0 "+u"\u2212"+" 8.0 Å")
patch = mpatches.Patch(facecolor=clrs[2], alpha=0.1,edgecolor=clrs[2],label='Pre-flare')
patch2 = mpatches.Patch(facecolor=clrs[6], alpha=0.1,edgecolor=clrs[6],label='Post-impulsive')
handles.append(line)
handles.append(patch)
handles.append(patch2)

leg = f.legend(handles=handles, bbox_to_anchor=(0.44, 0.88), bbox_transform=plt.gcf().transFigure, edgecolor="None",facecolor='w',framealpha=1)

plt.savefig("goes_plot.pdf", dpi=400, bbox_inches="tight")
