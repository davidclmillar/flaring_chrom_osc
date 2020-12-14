import numpy as np
import pickle
from astropy.io import fits
import sunpy.map as mp
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from reproject import reproject_interp
import matplotlib.colors as clr
from matplotlib import cm
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
import seaborn as sns
palette = sns.set_palette("colorblind")
clrs = sns.color_palette(palette)
from matplotlib import colors
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

# load a crisp fits file and alter header for the isolated wavelength point
crisp_file = "crisp_l2_20140906_152724_8542_r00470.fits"
crisp_fits = fits.open(crisp_file)
header = crisp_fits[0].header

bl_x = header['CRVAL1'] + (0 - header['CRPIX1'])*header['CDELT1']
bl_y = header['CRVAL2'] + (0 - header['CRPIX2'])*header['CDELT2']

header['NAXIS']=2
header['NAXIS1']=147
header['NAXIS2']=193
header['CDELT1']=0.57
header['CDELT2']=0.57
header['CRVAL1']=bl_x
header['CRVAL2']=bl_y
header['CRPIX1']=0
header['CRPIX2']=0
header['PC1_2']=0.0
header['PC2_1']=0.0
header['DATE-OBS']=header['DATE-AVG']

del header['NAXIS3']
del header['CDELT3']
del header['CRPIX3']
del header['CRVAL3']
del header['CUNIT3']
del header['WSTART1']
del header['WWIDTH1']
del header['WDESC1']
del header['TWAVE1']
del header['CTYPE3']
del header['PC3_3']

#load preferred models results and turn into a map object
pref_data1 = pickle.load(open("pref_ca8542_10_post.p","rb"))
pref_data2 = pickle.load(open("pref_ca8542_10_pre.p","rb"))
pref_data1[pref_data1!=2]=0
pref_data2[pref_data2!=2]=0
pref_map1 = mp.Map(pref_data1,header)
pref_map2 = mp.Map(pref_data2,header)

# similar as above but for an intensity image for the footpoints
foot_data = np.load("ca8542_12_post.npy")
foot_data = foot_data[0,:,:]
foot_data = foot_data/np.max(foot_data)
footheader = header


# make it a map
foot_map = mp.Map(foot_data,footheader)


#load in wing data for umbra lines
um_data = np.load("cube_ca8542_00_post_0.1res.npy")
um_data = um_data[0,:,:]
um_data = um_data/np.max(um_data)
umheader = header

um_map = mp.Map(um_data,umheader)


# load in aia submap
aia_map1 = mp.Map("pre_171x.fits")
aia_map2 = mp.Map("post_171x.fits")

# reproject pref, foot and hmi onto aia plane
#pref
y1, footprint = reproject_interp((pref_data1, header), aia_map1.wcs, aia_map1.data.shape)
#clean up pref result
y1 = np.nan_to_num(y1,0)
y1[y1>0.9]=1
y1[y1!=1]=np.nan

y2, footprint = reproject_interp((pref_data2, header), aia_map1.wcs, aia_map1.data.shape)
#clean up pref result
y2 = np.nan_to_num(y2,0)
y2[y2>0.9]=1
y2[y2!=1]=np.nan

#wing
outputum, footprintum = reproject_interp((um_data, umheader), aia_map1.wcs, aia_map1.data.shape)
#foot
footout, footfootprint = reproject_interp((foot_data,footheader), aia_map1.wcs, aia_map1.data.shape)
#make into maps
out_pref1 = mp.Map(y1,aia_map1.wcs)
out_pref2 = mp.Map(y2,aia_map1.wcs)
out_um = mp.Map(outputum,aia_map1.wcs)
out_foot = mp.Map(footout, aia_map1.wcs)

#----------------------
# --- make combined plot!
#----------------------

aiacmap = pickle.load(open("aia171cmap.p","rb"))
contourcolors = colors.ListedColormap('w')
footcolors =  colors.ListedColormap(clrs[2])
footplt = out_foot.data
footplt = np.nan_to_num(footplt,0)

# umbra contours
cont = out_um.data
cont=np.nan_to_num(cont,0)
conts = cont/np.max(cont)
m = np.zeros_like(conts)
m[60:-70,60:-70]=1
conts=m*conts
conts[conts==0]=np.nan

im1711 = aia_map1.data
im1712 = aia_map2.data
clip = 15 #no. of clip pixels off the edges
scale = 0.599489 # arcsec per pix
reduction = scale*clip

# this extent lines all features up with previous plots
extent = [-786.7816 + reduction, -667.9816 - reduction, -370.2548 + reduction, -250.8548 - reduction]

new_pref1 = out_pref1.data
new_pref2 = out_pref2.data
prefcolors = colors.ListedColormap([clrs[4],clrs[4]])


# start plotting
f, [ax1, ax2] = plt.subplots(1,2,figsize=(20,15))
im1711 = im1711[clip:-clip,clip:-clip]
im1712 = im1712[clip:-clip,clip:-clip]
new_pref1 = new_pref1[clip:-clip,clip:-clip]
new_pref2 = new_pref2[clip:-clip,clip:-clip]
conts = conts[clip:-clip,clip:-clip]
footplt = footplt[clip:-clip,clip:-clip]

ax1.imshow(np.log2(im1711),origin='lower',extent=extent,cmap=aiacmap,vmax=12)
ax1.imshow(new_pref2,origin='lower',cmap=prefcolors, extent=extent)
ax1.contour(conts, origin='lower', levels=[0.4], extent=extent, cmap=contourcolors)
ax1.contour(footplt/np.max(footplt),extent=extent,origin='lower',levels=[0.55],cmap=contourcolors,linestyles='--')
ax1.set_xlabel("Solar X [arcsec]")
ax1.set_ylabel("Solar Y [arcsec]")
ax1.xaxis.set_major_locator(MultipleLocator(20))
ax1.xaxis.set_minor_locator(MultipleLocator(10))
ax1.yaxis.set_major_locator(MultipleLocator(20))
ax1.yaxis.set_minor_locator(MultipleLocator(10))
ax1.set_title("AIA 171 \u212B, 16:36:35")

ax2.imshow(np.log2(im1712),origin='lower',extent=extent,cmap=aiacmap,vmax=12)
ax2.imshow(new_pref1,origin='lower',cmap=prefcolors, extent=extent)
ax2.contour(conts, origin='lower', levels=[0.4], extent=extent, cmap=contourcolors)
ax2.contour(footplt/np.max(footplt),extent=extent,origin='lower',levels=[0.55],cmap=contourcolors,linestyles="--")
ax2.set_xlabel("Solar X [arcsec]")
ax2.xaxis.set_major_locator(MultipleLocator(20))
ax2.xaxis.set_minor_locator(MultipleLocator(10))
ax2.yaxis.set_major_locator(MultipleLocator(20))
ax2.yaxis.set_minor_locator(MultipleLocator(10))
ax2.set_title("AIA 171 \u212B, 17:26:36")
plt.show()
