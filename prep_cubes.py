import numpy as np
import os
from astropy.io import fits

def rebin(im,factor):
    dims = im.shape
    if (dims[0] % factor == 0) and (dims[1] % factor == 0):
        newdims = (int(dims[0]/factor),int(dims[1]/factor))
        im_new = np.zeros(shape=newdims)
        for x in range(newdims[0]):
            for y in range(newdims[1]):
                im_new[x,y] = np.sum(im[factor*x:factor*(x+1),factor*y:factor*(y+1)])    
    else:
        raise ValueError ("factor must divide dimensions exactly")
        
    return im_new

# -- do many crisp files -- #
for line in ["Halpha","ca8542"]:

    directory = "crisp_data/%s/"%line

    filelist = sorted(os.listdir(directory))

    if line=="ca8542":
        wls=np.arange(25)
        no="8542"
    elif line=='Halpha':
        wls=np.arange(15)
        no="6563"
    else:
        raise ValueError("bad line name")

    for wl in wls:
        print("\r %s %d  "%(line,wl),end="")
    # pre
        cube = np.zeros(shape=(148,139,147))
        for i in range(253,401):
            file = os.path.join(directory,"crisp_l2_20140906_152724_%s_r00%s.fits"%(no,str(i).zfill(3)))
            print("\r"+file,end="")
            hdulist = fits.open(os.path.join(directory, file))
            im_old  = hdulist[0].data[wl,:1390,:1470] # clip the edges so we have round numbered dimensions
            hdulist.close()
            im_new  = rebin(im_old,10)
            cube[i-253,:,:]=im_new
            del im_old, im_new, file
        savedir = "rebinned_by_10/%s/pre/"%line
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.save(savedir+"%s_%s_pre"%(line, str(wl).zfill(2)), cube)
    # post
        cube = np.zeros(shape=(148,139,147))
        for i in range(470,618):
            file = os.path.join(directory,"crisp_l2_20140906_152724_%s_r00%s.fits"%(no,str(i).zfill(3)))
            print("\r"+file,end="")
            hdulist = fits.open(os.path.join(directory, file))
            im_old  = hdulist[0].data[wl,:1390,:1470]
            hdulist.close()
            im_new  = rebin(im_old,10)
            cube[i-470,:,:]=im_new
            del im_old, im_new
        savedir = "rebinned_by_10/%s/post/"%line
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.save(savedir+"%s_%s_post"%(line, str(wl).zfill(2)), cube)
