import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm
import kappa_fitting
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
import pickle
import seaborn as sns
palette = sns.set_palette("colorblind")
clrs = sns.color_palette(palette)
from matplotlib import colors
from matplotlib import gridspec
from astropy.io import fits
import matplotlib.ticker as ticker
# get the bottom left of images x and y coords
hdu = fits.open("crisp_l2_20140906_152724_8542_r00470.fits")
header = hdu[0].header
bl_x = header['CRVAL1'] + (0 - header['CRPIX1'])*header['CDELT1']
bl_y = header['CRVAL2'] + (0 - header['CRPIX2'])*header['CDELT2']
    
print(bl_x,bl_y)


# ---- set up contours  ---- #
one = pickle.load(open("pref_ca8542_12_post.p","rb"))

cont = np.load("cube_wl_00.npy")
cont = cont[0,:,:]
cont = cont/np.max(cont)
cmap = colors.ListedColormap([clrs[0],clrs[4],clrs[8]])

contourcolors = colors.ListedColormap(['w'])
levels = np.array([0.4,0.75])

conts = np.ones_like(cont)
conts[cont<levels[0]]=0
conts[cont>levels[1]]=2
conts[one==0]=np.nan

line="Halpha"
time = 'post'
if line=="ca8542":
    wls = [0,4,6,8,10,12,14,16,18,24]
    rows,cols = 2,5
    d_wl = 0.1
    wl0 = 12
    title = "Ca II 8542 \u212B %s-impulsive, preferred models"%(time)
elif line=="Halpha":
    wls = [1,3,4,5,6,7,8,9,10,13]
    rows,cols=2,5
    d_wl = 0.2
    wl0 = 7
    title = "H-alpha %s-impulsive, preferred models"%(time)
# ---- multiple plots on a gridspec ---- #
sizefactor = 3
# - set up figure and grids
f = plt.figure(figsize = (cols*sizefactor,rows*sizefactor))
f.suptitle(title, y = 0.95, fontsize=20)
gs1 = gridspec.GridSpec(rows, cols,wspace=1,hspace=1)
gs1.tight_layout(f, renderer=None, pad=1, h_pad=1, w_pad=1)
# clip edges
clip_size = 5
scale = 0.57 # arcsec per pixel
conts = conts[clip_size:-clip_size,clip_size:-clip_size]

# loop through our wls
for a in range(rows*cols):
    wl = wls[a]
    ax1 = plt.subplot(gs1[a])
    path = "folder/pref_%s_%s_%s.p"%(line,str(wl).zfill(2),time)

    pref = pickle.load(open(path,"rb"))
    pref[pref==0]=np.nan
    pref = pref[clip_size:-clip_size,clip_size:-clip_size]
    # find extent
    bottom = bl_y + clip_size*scale
    top = bottom + pref.shape[0]*scale
    left = bl_x + clip_size*scale
    right = left + pref.shape[1]*scale

    extent = (left, right, bottom, top)
    # add a new plot to gridspec
    im = plt.imshow(pref,origin='lower',cmap=cmap,extent=extent)
    plt.contour(conts,origin='lower',levels=[0,1,2,3],cmap=contourcolors,linewidths=(1,0.1,0.1,0.1),linestyles=[':','-'],extent=extent,alpha=0.7)
    plt.axis('on')
    # set up ticks
    k = cols*(rows - 1)- 1

    ax1.tick_params(labelsize=11)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    #ax1.set_xticklabels([])
    #ax1.set_yticklabels([-340,"","","","","",-290])

    if a % cols > 0 and a <= k:
        ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)#, direction='in')
    elif a % cols > 0 and a >= k:
        ax1.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False  , labelleft=False)#, direction='in')
    elif a % cols == 0 and a < k:
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False   , labelleft=False)#, direction='in')
    elif a % cols == 0 and a > k :
        ax1.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, right=False, left=True   , labelleft=True)# , direction='in')

    m = rows*cols - np.ceil(cols/2)
    n = int(rows/2)*cols
    #labels
    if a == m:
        ax1.set_xlabel('Solar X [arcsec]')#,fontsize=12)
    if a == n:
        ax1.set_ylabel('Solar Y [arcsec]')#,fontsize=12)
    ax1.set_aspect('auto')
    # wavelength labels
    real_wl = round(d_wl*(wls[a]-wl0),1)
    text_x = -715
    text_y = -290
    rot_angle = 0
    
    if real_wl > 0:
        ax1.text(text_x,text_y,"$+$"+str(real_wl)+"\u212B", rotation=rot_angle,fontsize=12)
    elif real_wl==0:
        ax1.text(text_x,text_y,"$\pm$"+str(real_wl)+"\u212B", rotation=rot_angle,fontsize=12)
    else:
        ax1.text(text_x,text_y,"$-$"+str(-real_wl)+"\u212B", rotation=rot_angle,fontsize=12)
    
    gs1.update(wspace=0.00, hspace=0.00) 


# - add cbar
cax = f.add_axes([0.9, 0.25, 0.02, 0.5])
upper = 3
lower = 1
N = 3

norm = clr.Normalize()

mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

deltac = (upper-lower)/(2*(N-1))

mapper.set_array(np.linspace(lower-deltac,upper+deltac,10)) #<-- the 10 here is pretty arbitrary
clb = f.colorbar(mapper, shrink=0.19, cax=cax,ticks=[1,2,3])
clb.ax.set_yticklabels(['M1','M2','M3'])
clb.ax.tick_params(size=0)#rotation=270,size=0)
#plt.show()




plt.savefig("preferred_%s_%s.pdf"%(line,time),dpi=400,bbox_inches='tight')
