import sunpy.map as mp
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
plt.rcParams["font.family"] = "serif"
from astropy.io import fits
import datetime

path = 'path_to_crisp_data'
i = 470 # which crisp file
filename = "crisp_l2_20140906_152724_8542_r00%s.fits"%(str(i).zfill(3))

hdulist = fits.open(path+filename)
header = hdulist[0].header
strorig = header['DATE-AVG']

# get the time of obs
dt = datetime.datetime.strptime(strorig,'%Y-%m-%dT%H:%M:%S.%f')
strnew = datetime.datetime.strftime(dt, '%H:%M:%S')
# two separate images
wingimage = hdulist[0].data[0 ,:,:]
coreimage = hdulist[0].data[12,:,:]

xcentrepix = header['CRPIX1']
xcentreval = header['CRVAL1']
secperpix  = 0.057
ycentrepix = header['CRPIX2']
ycentreval = header['CRVAL2']


im = coreimage
mode = im[0,0]

# create "mask"
mask = np.ones_like(im)

for x in range(im.shape[0]):
    print("\r"+str(x),end="")
    y = 0
    while (im[x,y] == mode) and (y<im.shape[1]-1):
    #while (y<im.shape[1]):
        mask[x,y] = 0
        y+=1

for x in range(im.shape[0]):
    print("\r"+str(x),end="")
    y = im.shape[1]-1
    while (im[x,y] == mode) and (y>0):
    #while (y<im.shape[1]):
        mask[x,y] = 0
        y-=1

mask = mask.astype(float)

header = hdulist[0].header
xcentrepix = header['CRPIX1']
xcentreval = header['CRVAL1']
secperpix  = 0.057
ycentrepix = header['CRPIX2']
ycentreval = header['CRVAL2']

# get extent
left  = xcentreval - xcentrepix*secperpix
right = xcentreval - (xcentrepix - coreimage.shape[1])*secperpix

bottom  = ycentreval - ycentrepix*secperpix
top = ycentreval - (ycentrepix - coreimage.shape[0])*secperpix

# load aia data and submap
aiamap = mp.Map("aia_lev1_1700a_2014_09_06t16_58_06_71z_image_lev1.fits")

bl_Tx,bl_Ty,tr_Tx,tr_Ty = left, bottom, right, top

bottom_left = SkyCoord(bl_Tx*u.arcsec, bl_Ty*u.arcsec, frame=aiamap.coordinate_frame)
top_right   = SkyCoord(tr_Tx*u.arcsec, tr_Ty*u.arcsec, frame=aiamap.coordinate_frame)
aiasbmap = aiamap.submap(bottom_left, top_right)

# submap for HMI
hmimap = mp.Map("hmi_20140906_165736_continuum.fits")
hmisbmap = hmimap.submap(bottom_left, top_right)
hmisbmap.rotate(angle=180*u.degree)


# -------------------------------------------------------------------------------------------
# ----------------   plot!  ---------------------------------------------------------
# ------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(2,2,1, projection=hmisbmap)
hmisbmap.plot()
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
ax1.set_xlabel('')
ax1.set_ylabel('Solar Y [arcsec]')
ax1.set_title('HMI continuum 16:57:36')

ax2 = fig.add_subplot(2,2,2, projection=aiasbmap)
aiasbmap.plot()
ax2.set_xlabel('')
ax2.set_ylabel("")
ax2.set_yticklabels([])
ax2.set_title('AIA 1700 \u212B 16:58:06')

xticks = [-760,-740,-720,-700]
yticks = [-340,-320,-300,-280]
xticklabels = ['-760.0','-740.0','-720.0','-700.0']
yticklabels = ['-340.0','-320.0','-300.0','-280.0']

maskedwing = wingimage*mask
maskedwing[maskedwing == 0] = np.nan



ax3 = fig.add_subplot(2,2,3)
ax3.imshow(maskedwing,origin='lower',extent=[left,right,bottom,top],cmap='gray')
ax3.set_ylabel('Solar Y [arcsec]')
ax3.set_xlabel('Solar X [arcsec]')
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_yticklabels(yticklabels)
ax3.set_xticklabels(xticklabels)
ax3.set_title('Ca II 8542 -1.2\u212B '+strnew)

maskedcore = coreimage*mask
maskedcore[maskedcore == 0] = np.nan

ax4 = fig.add_subplot(2,2,4)
ax4.imshow(np.log2(maskedcore),origin='lower',extent=[left,right,bottom,top],cmap='gray')
ax4.set_ylabel('')
ax4.set_xlabel('Solar X [arcsec]')
ax4.set_xticks(xticks)
ax4.set_yticks(yticks)
ax4.set_yticklabels(yticklabels)
ax4.set_xticklabels(xticklabels)
ax4.set_title('Ca II 8542 \u00B10.0\u212B '+strnew)


plt.savefig("context_images.pdf",dpi=400,bbox_inches='tight')

