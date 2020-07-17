import numpy as np
import pickle
from astropy.io import fits
import sunpy.map as mp
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
crisp_fits = fits.open("file_load")
header = crisp_fits[0].header
crisp_fits.close()

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
pref_data = pickle.load(open("folder/pref_ca8542_10_post.p","rb"))
pref_data[pref_data!=2]=0
pref_map = mp.Map(pref_data,header)


# similar as above but for an intensity image for the footpoints
foot_fits = fits.open("")
footheader = foot_fits[0].header
footdata   = foot_fits[0].data[12,:-8,:-3]

footheader['NAXIS']=2
footheader['PC1_2']=0.0
footheader['PC2_1']=0.0
footheader['DATE-OBS']=footheader['DATE-AVG']

del footheader['NAXIS3']
del footheader['CDELT3']
del footheader['CRPIX3']
del footheader['CRVAL3']
del footheader['CUNIT3']
del footheader['WSTART1']
del footheader['WWIDTH1']
del footheader['WDESC1']
del footheader['TWAVE1']
del footheader['CTYPE3']
del footheader['PC3_3']

# make it a map
foot_map = mp.Map(footdata,footheader)

# load in hmi submap
hmi_map = mp.Map("location/hmi172630submap.fits")
hmi_fits = fits.open("location/hmi172630submap.fits")
# load in aia submap
aia_map = mp.Map("location/aia172630submap.fits")

# plot to see originals
#f = plt.figure(figsize=(15,15))
#ax1 = f.add_subplot(2,2,1,projection = aia_map)
#ax2 = f.add_subplot(2,2,2,projection = hmi_map)
#ax3 = f.add_subplot(2,2,3,projection = pref_map)
#ax4 = f.add_subplot(2,2,4,projection = foot_map)
#aia_map.plot(axes=ax1)
#hmi_map.plot(axes=ax2)
#pref_map.plot(axes=ax3)
#foot_map.plot(axes=ax4)
#plt.show()

# reproject pref, foot and hmi onto aia plane
#pref
y, footprint = reproject_interp((pref_data,header), aia_map.wcs, aia_map.data.shape)
#clean up pref result
y = np.nan_to_num(y,0)
y[y>0.9]=1
y[y!=1]=np.nan
#hmi
outputhmi, footprinthmi = reproject_interp((hmi_fits[0].data, hmi_fits[0].header), aia_map.wcs, aia_map.data.shape)
#foot
footout, footfootprint = reproject_interp((footdata,footheader), aia_map.wcs, aia_map.data.shape)
#make into maps
out_pref = mp.Map(y,aia_map.wcs)
out_hmi = mp.Map(outputhmi,aia_map.wcs)
out_foot = mp.Map(footout, aia_map.wcs)

#plot reprojections
#print("Reprojected images")

#f = plt.figure(figsize=(15,15))
#ax1 = f.add_subplot(2,2,1,projection = aia_map)
#ax2 = f.add_subplot(2,2,2,projection = out_hmi)
#ax3 = f.add_subplot(2,2,3,projection = out_pref)
#ax4 = f.add_subplot(2,2,4,projection = out_foot)
#aia_map.plot(axes=ax1)
#out_hmi.plot(axes=ax2)
#out_pref.plot(axes=ax3)
#out_foot.plot(axes=ax4)
#plt.show()

#make combined plot!

aiacmap = "your_fave_cmap"
contourcolors = colors.ListedColormap(clrs[3])
footcolors =  colors.ListedColormap(clrs[2],clrs[2])
footplt = out_foot.data
footplt = np.nan_to_num(footplt,0)

# umbra contours
levels=[0.4]
cont = out_hmi.data
cont=np.nan_to_num(cont,0)
cont = cont/np.max(cont)
conts = np.ones_like(cont)
conts[cont<levels[0]]=0

im171 = aia_map.data
clip = 15 #no. of clip pixels off the edges
scale = 0.599489 # arcsec per pix
reduction = scale*clip

# this extent lines all features up with previous plots
extent = [-786.7816 + reduction, -667.9816 - reduction, -370.2548 + reduction, -250.8548 - reduction]

new_pref = out_pref.data
prefcolors = colors.ListedColormap([clrs[4],clrs[4]])

f, ax1 = plt.subplots(figsize=(12,12))
im171 = im171[clip:-clip,clip:-clip]
new_pref = new_pref[clip:-clip,clip:-clip]
conts = conts[clip:-clip,clip:-clip]
footplt = footplt[clip:-clip,clip:-clip]

plt.imshow(np.log2(im171),origin='lower',extent=extent,cmap=aiacmap,vmin=7.5,vmax=11)
plt.imshow(new_pref,origin='lower',cmap=prefcolors, extent=extent)
#plt.colorbar()
plt.contour(conts, origin='lower', levels=[0,1,2], extent=extent, cmap=contourcolors)
plt.contourf(footplt/np.max(footplt),extent=extent,origin='lower',levels=[0.7,1],cmap=footcolors,alpha=0.8)
ax1.set_xlabel("Solar X [arcsec]")
ax1.set_ylabel("Solar Y [arcsec]")
ax1.xaxis.set_major_locator(MultipleLocator(20))
ax1.xaxis.set_minor_locator(MultipleLocator(10))
ax1.yaxis.set_major_locator(MultipleLocator(20))
ax1.yaxis.set_minor_locator(MultipleLocator(10))

plt.title("AIA 171 \u212B, 17:26:36")
plt.show()
plt.savefig("combine.pdf",dpi=400,bbox_inches='tight')
