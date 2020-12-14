import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
import pickle
import seaborn as sns
palette = sns.set_palette("colorblind")
clrs = sns.color_palette(palette)
from matplotlib import colors
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from astropy.io import fits

# set contours
contourcolors = colors.ListedColormap(['w','w','w','w'])
levels=[0.3,0.73]
hmisubmap = fits.open("submap.fits")
cont = hmisubmap[0].data
cont = np.flip(cont,axis=(0,1))
cont = cont/np.max(cont)
conts = np.ones_like(cont)
conts[cont<levels[0]]=0
conts[cont>levels[1]]=2

# set up figure
rows,cols=2,2
sizefactor=4
f = plt.figure(figsize = (cols*sizefactor,rows*sizefactor))
# shidts to apply due to solar rotation, calculated from sunpy solar rotation package
x_shift = -3.7184
y_shift =  0.9548

# original crisp information for the bottom left x and y in arcsecs
hdu = fits.open("crisp_l2_20140906_152724_8542_r00470.fits")
header = hdu[0].header
bl_x = header['CRVAL1'] + (0 - header['CRPIX1'])*header['CDELT1']
bl_y = header['CRVAL2'] + (0 - header['CRPIX2'])*header['CDELT2']


# read in aia
aia1hdu = fits.open("aia_submap_1700a_2014_09_06t16_14_06_72z_image_lev1.fits")
header=aia1hdu[0].header
aia1_bl_x = header['CRVAL1'] + (0 - header['CRPIX1'])*header['CDELT1']
aia1_bl_y = header['CRVAL2'] + (0 - header['CRPIX2'])*header['CDELT2']

# for extent of images
aia_bl_x = aia1_bl_x - x_shift
aia_bl_y = aia1_bl_y - y_shift

files = ["aia_pref/%s_%s_pref.p"%(wl, time) for wl,time in zip(["1600","1600","1700","1700"],["pre","post","pre","post"])]

gs1 = gridspec.GridSpec(rows, cols,wspace=0.15,hspace=0.15)
gs1.tight_layout(f, renderer=None, pad=1, h_pad=1, w_pad=1)
cmap = colors.ListedColormap([clrs[0],clrs[4],clrs[8]])

clip_size=10
scale = 0.6
clip = clip_size*scale # this is in arcsec and is the same for both

hmi_scale = 0.504
clip_size_hmi = round(clip/0.504)

#clip edges off
conts= conts[clip_size_hmi:,0:-clip_size_hmi]

# load in each and plot on grid
for a in range(rows*cols):
    ax1 = plt.subplot(gs1[a])
    path=files[a]
    pref = pickle.load(open(path,"rb"))
    pref[pref==0]=np.nan
    pref = pref[clip_size:,0:-clip_size] #clip off bottom and right
    left   = aia_bl_x
    right  = aia_bl_x + scale*pref.shape[0]
    bottom = aia_bl_y + clip
    top    = bottom + scale*pref.shape[1]

    left_hmi   = aia_bl_x
    right_hmi  = aia_bl_x + hmi_scale*conts.shape[0]
    bottom_hmi = aia_bl_y + clip
    top_hmi    = bottom + hmi_scale*conts.shape[1]

    extent = [left,right,bottom,top]
    extent_hmi = [left_hmi, right_hmi, bottom_hmi, top_hmi]
    plt.imshow(pref,origin='lower',cmap=cmap,extent=extent)
    plt.contour(conts,origin='lower',levels=[0,1,2,3],cmap=contourcolors,linewidths=0.6,linestyles=[':','-'],extent=extent_hmi)

    # ticks
    k = cols*(rows - 1)- 1

    if a % cols > 0 and a <= k:
        ax1.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=False, right=False, left=True, labelleft=False, direction='out')
    elif a % cols > 0 and a >= k:
        ax1.tick_params(axis='y', which='major', bottom=True, top=False, labelbottom=False, right=False, left=True  , labelleft=False, direction='out')
    elif a % cols == 0 and a < k:
        ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False, right=False, left=True   , labelleft=False, direction='out')
    elif a % cols == 0 and a > k :
        ax1.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, right=False, left=True   , labelleft=True , direction='out')

    ax1.tick_params(labelsize=10)

    # labels
    m = rows*cols - int(cols/2)
    n = int(rows/2)*cols
    if a == m:
        ax1.set_xlabel('Solar X [arcsec]',fontsize=11)
    if a == n:
        ax1.set_ylabel('Solar Y [arcsec]',fontsize=11)
    ax1.set_aspect('auto')
    ax1.xaxis.set_major_locator(MultipleLocator(20))
    ax1.xaxis.set_minor_locator(MultipleLocator(10))

    ax1.yaxis.set_major_locator(MultipleLocator(20))
    ax1.yaxis.set_minor_locator(MultipleLocator(10))

    blip = path[18:-7]
    if blip=="pre":

        ax1.set_title(path[13:17]+u" \u212B "+blip +"-flare",color='k',fontsize=12)
    else:
        ax1.set_title(path[13:17]+u" \u212B "+blip +"-impulsive",color='k',fontsize=12)

# - add cbar
cax = f.add_axes([0.92, 0.25, 0.02, 0.5])
upper = 3
lower = 1
N = 3

norm = clr.Normalize()

mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

deltac = (upper-lower)/(2*(N-1))

mapper.set_array(np.linspace(lower-deltac,upper+deltac,10))
clb = f.colorbar(mapper, shrink=0.19, cax=cax,ticks=[1,2,3])
clb.ax.set_yticklabels(['M1','M2','M3'])
clb.ax.tick_params(size=0)#rotation=270,size=0)
#plt.show()
plt.savefig("aia_prefer.pdf",dpi=400,bbox_inches='tight')


