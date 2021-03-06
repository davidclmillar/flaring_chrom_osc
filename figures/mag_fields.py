import numpy as np
import pickle
from astropy.io import fits
import sunpy.map as mp
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

import seaborn as sns
palette = sns.set_palette("colorblind")
clrs = sns.color_palette(palette)
clrs[5]=clrs[8]
from astropy.coordinates import SkyCoord as sc
import astropy.units as u
import os
import matplotlib.dates as dates


from datetime import datetime

%matplotlib inline
os.chdir("../data/sharp/")

blX, blY = 91,-15.5
trX, trY = 94.2,-13.5
extent = [blX, trX, blY, trY]

# diff = mp.Map(os.path.join(folder,files[7]))
# bldiff = SkyCoord(blX*u.degree,blY*u.degree,frame=diff.coordinate_frame)
# trdiff = SkyCoord(trX*u.degree,trY*u.degree,frame=diff.coordinate_frame)

# sdiff = diff.submap(bldiff,trdiff)

# diffdata = sdiff.data

cont = mp.Map("continuum/hmi.sharp_cea_720s.4530.20140906_170000_TAI.continuum.fits")
blcont = sc(blX*u.degree,blY*u.degree,frame=cont.coordinate_frame)
trcont = sc(trX*u.degree,trY*u.degree,frame=cont.coordinate_frame)
subcont = cont.submap(blcont,trcont)
normcont = subcont.data/np.max(subcont.data)

points = np.zeros(shape=(2,9))

flarestart = datetime(2014,9,6,16,47,0,0)
flarepeak  = datetime(2014,9,6,17, 9,0,0)
size=1

markers = ["o", "v", "^", "s", "P", "p", "D","X"]
plt.rcParams["font.size"] = 14
for f in range(0,1):
    
    points = np.zeros(shape=(2,8))
    
    points[:,0] = [-14.75, 92.5]
    points[:,1] = [-14.60, 92.1]
    points[:,2] = [-14.50, 92.56]
    points[:,3] = [-14.60, 92.80]
    points[:,4] = [-14.40, 92.5]
    points[:,5] = [-14.40, 92.80]
    points[:,6] = [-14.75, 92.20]
    points[:,7] = [-14.40, 92.1 ]
    #points[:,8] = [, ]
    
    
    points = points + f
    print(points)
    plt.figure(figsize=(15,15))
    ax = plt.subplot(2,2,1)
    plt.imshow(normcont, origin='lower',cmap='Greys_r',extent=extent)
    plt.xlabel('Carrington Longitude [deg]',fontsize=14)
    plt.ylabel('Latitude [deg]',fontsize=14)
    plt.title('HMI Continuum',fontsize=16)
    for j in range(points.shape[1]):
        plt.scatter(points[1,j],points[0,j],color=clrs[j],marker=markers[j],s=55)
        ax.text(points[1,j]+0.02,points[0,j]+0.03,str(j+1),color=clrs[j])
    ax.set_aspect(1.75)
    segs = ["Br","Bp","Bt"]
    

    segfancy = [r"$B_r$",r"$B_\phi$",r"$B_\theta$"]
    
    for counter, seg in enumerate(segs):
        folder = "./%s"%seg
        files = sorted(os.listdir(folder))
        files = [j for j in files if ".fits" in j]

        timesrs = np.zeros(shape=(points.shape[1],len(files)))
        #print(timesrs.shape)
        
        timestamps = []
        for i in range(len(files)):
            x = mp.Map(os.path.join(folder,files[i]))
            bl = sc(blX*u.degree,blY*u.degree,frame=x.coordinate_frame)
            tr = sc(trX*u.degree,trY*u.degree,frame=x.coordinate_frame)
            s = x.submap(bl,tr)
            timestamps.append(datetime.strptime(s.meta['t_rec'],"%Y.%m.%d_%H:%M:%S_TAI"))
            ss = s.data

            for p in range(points.shape[1]):
                x = points[1,p]*u.degree
                y = points[0,p]*u.degree
                co = sc(x,y,frame=s.coordinate_frame)
                r = s.world_to_pixel(co,origin=1)
                row = round(r[1].value)
                col = round(r[0].value)
                value = np.sum(ss[row:row+size, col:col+size])/size**2
                timesrs[p,i]=value
        
        #timesrs = np.abs(timesrs)
#         timesrs = np.mean(timesrs,axis=1)
        plot_times = dates.date2num(timestamps)
        myFmt = dates.DateFormatter('%H:%M')
        ax = plt.subplot(2,2,counter+2)
        for c in range(timesrs.shape[0]):
            t = timesrs[c,:]
            t = np.abs(t)
            #t= t/np.max(t)
            #t=t-c*0.2
            plt.plot_date(timestamps, t, color=clrs[c], linestyle='-',marker=markers[c],markersize=10)
        ax.xaxis.set_major_formatter(myFmt)
        plt.title(segfancy[counter],fontsize=16)
        plt.xlabel('Time',fontsize=14)
        plt.ylabel("|"+segfancy[counter]+"| [G]",fontsize=14)
        ax.axvline(flarestart,color='k',linestyle='--',linewidth='0.5')
        ax.axvline(flarepeak,color='k',linestyle='--',linewidth='0.5')
        #plt.legend()
        #plt.savefig()
