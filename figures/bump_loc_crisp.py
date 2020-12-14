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
from astropy.io import fits
import matplotlib.ticker as ticker

# get the bottom left of images x and y coords
hdu = fits.open("crisp_l2_20140906_152724_8542_r00470.fits")
header = hdu[0].header
bl_x = header['CRVAL1'] + (0 - header['CRPIX1'])*header['CDELT1']
bl_y = header['CRVAL2'] + (0 - header['CRPIX2'])*header['CDELT2']
hdu.close()

# ---- set up contours for umbra  ---- #
one = pickle.load(open("pref_results.p","rb"))

cont = np.load("cube_ca8542_00_post_0.1res.npy")
cont = cont[0,:,:]
cont = cont/np.max(cont)
cmap = colors.ListedColormap([clrs[0],clrs[4],clrs[8]])

contourcolors = colors.ListedColormap(['gray','gray','gray','gray'])
levels = np.array([0.4,0.75])

conts = np.ones_like(cont)
conts[cont<levels[0]]=0
conts[cont>levels[1]]=2
conts[one==0]=np.nan # pixels out of FOV set to nan

#  ----- flare feet contours - added 20201202 ----- #

contf = np.load("ca8542_12_post.npy")
contf = contf[0,:,:]
contf = contf/np.max(contf)
footcolors = colors.ListedColormap(clrs[6])
contf[one==0]=np.nan

cmap = 'viridis'

line="ca8542"
time = 'post'
if line=="ca8542":
    wls = [8,10,12,14,16] # can change
    rows,cols = 1,5
    d_wl = 0.1
    wl0 = 12
    title = "Ca II 8542 \u212B %s-impulsive, Gaussian bump peaks"%(time)
elif line=="Halpha":
    wls = [5,6,7,8,9]
    rows,cols=1,5
    d_wl = 0.2
    wl0 = 7
    title = "H\u03B1 %s-impulsive, Gaussian bump peaks"%(time)

# ---- multiple plots on a gridspec ---- #
sizefactor = 3
# - set up figure and grids
f = plt.figure(figsize = (cols*sizefactor,rows*sizefactor))

f.suptitle(title, y = 0.98, fontsize=16)

gs1 = gridspec.GridSpec(rows, cols,wspace=1,hspace=1)
gs1.tight_layout(f, renderer=None, pad=1, h_pad=1, w_pad=1)
# clip edges to focus on sunspot
clip_left   = 10
clip_right  = 40
clip_bottom = 10
clip_top    = 40
scale=0.57
conts = conts[clip_top:-clip_bottom,clip_left:-clip_right]
contf = contf[clip_top:-clip_bottom,clip_left:-clip_right]

# loop through wavelength points and add each to the gridspec
for a in range(rows*cols):
    wl = wls[a]
    ax1 = plt.subplot(gs1[a])
    path_bump = "parameter_results.p"%(line,str(wl).zfill(2),time)
    path_pref = "pref_results.p"%(line,str(wl).zfill(2),time)
    loaded_results = pickle.load(open(path_bump,"rb"))
    loaded_pref = pickle.load(open(path_pref,"rb"))


    bump = loaded_results[1][3,:,:] # this isolates the map of beta parameters


    bump[loaded_pref!=2]=np.nan

    bump = bump[clip_top:-clip_bottom,clip_left:-clip_right]

   # bottom = bl_y + clip_bottom*scale
    bottom = bl_y + clip_top*scale
    top = bottom + bump.shape[0]*scale
    left = bl_x + clip_left*scale
    right = left + bump.shape[1]*scale

    extent = (left, right, bottom, top)

    bump = np.e**(-bump) # convert to period (s)
    plt.contour(conts,origin='lower',levels=[0,1,2,3],cmap=contourcolors,linewidths=0.6,linestyles=[':','-'],extent=extent)
    plt.contourf(contf,extent=extent,origin='lower',levels=[0.55,1],cmap=footcolors,alpha=0.5)
    #plt.contour(contf,extent=extent,origin='lower',levels=[0.55,1],cmap=footcolors,linewidths=0.8,linestyles=['-.'])
    im = plt.imshow(bump, origin='lower', cmap=cmap, extent=extent, vmin=100, vmax = 300)
    
    plt.axis('on')
#---------------------------
    # set up ticks, labels
#---------------------------
    k = cols*(rows - 1)- 1

    ax1.tick_params(labelsize=11)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(10))

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

    if a == m:
        ax1.set_xlabel('Solar X [arcsec]',fontsize=12)
    if a == n:
        ax1.set_ylabel('Solar Y [arcsec]',fontsize=12)
    ax1.set_aspect('auto')
#-----------------------
    # wavelength labels
#-----------------------
    real_wl = round(d_wl*(wls[a]-wl0),1)
    text_x = -762
    text_y = -290
    rot_angle = 0

    if real_wl > 0:
        ax1.text(text_x,text_y,"$+$"+str(real_wl)+"\u212B", rotation=rot_angle,fontsize=12)
    elif real_wl==0:
        ax1.text(text_x,text_y,"$\pm$"+str(real_wl)+"\u212B", rotation=rot_angle,fontsize=12)
    else:
        ax1.text(text_x,text_y,"$-$"+str(-real_wl)+"\u212B", rotation=rot_angle,fontsize=12)

    gs1.update(wspace=0.00, hspace=0.00) 

#---------------
# - add cbar
#---------------
cax = f.add_axes([0.9, 0.25, 0.02, 0.5])
upper = 3
lower = 1
N = 3
norm = clr.Normalize()
mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
deltac = (upper-lower)/(2*(N-1))
clb = plt.colorbar(cax=cax)
clb.set_label("Period [s]", rotation=270,labelpad=18,fontsize=12)
clb.set_ticks([100,200,300])
clb.set_ticklabels(['100','200','>300'])
plt.show()
