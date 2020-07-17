

import numpy as np
import matplotlib.pyplot as plt
import kappa_fitting
from kappa_fitting import timeseries
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
import matplotlib.dates as dates
minutes = dates.MinuteLocator()
from datetime import datetime,timedelta
import seaborn as sns
palette = sns.set_palette("colorblind")
clrs = sns.color_palette(palette)
# set time axis
start_time = datetime(2014,9,6,16,58,3,844000)
times = [start_time + timedelta(seconds=11.6*x) for x in range(148)]
x_vals = dates.date2num(times)
myFmt = dates.DateFormatter('%H:%M')

# choose macropixel
row, col = 85, 43
datacube= np.load("cube_ca8542_12_post_0.1res.npy")

s = datacube[:,row, col]
x = timeseries(s)

# run the fit
result = x.fit_M2(plot=False)
popt = result[0]


# plot
f, ax = plt.subplots(1,2,figsize=(12,4))
f.suptitle('Timeseries and PSD best described by M2. Ca II 8542 $\pm$0.0\u212B',y=0.98)


ax[0].plot_date(x_vals,x.normalised, linestyle='-',marker=None,color='k')#,color=clrs[2])
ax[0].set_xlabel('2014-09-06')
ax[0].set_ylabel(r'Intensity [$\sigma$]')
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].xaxis.set_minor_locator(minutes)
ax[0].set_title('')

m = kappa_fitting.siglvl(0.999, len(x.freqs))

modelk = kappa_fitting.model2
ax[1].step(1/x.freqs,x.psd,label='PSD')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel('Period [s]')
ax[1].set_ylabel(r'Power [$\sigma^2$]')
ax[1].plot(1/x.freqs, x.weights,label='GWS',linestyle='--')
ax[1].plot(1/x.freqs, modelk(x.freqs,*popt),label='M2 fit')
ax[1].plot(1/x.freqs,m*modelk(x.freqs,*(popt*[1,1,0,1,1,1])),label='99.9%',linestyle=':')
ax[1].legend()

plt.subplots_adjust(wspace=0.25)
plt.savefig("gauss_example_new.pdf",dpi=400,bbox_inches='tight')

#plt.show()
