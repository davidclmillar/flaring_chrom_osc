import numpy as np
import pickle
import kappa_fitting
import os

# save folder
savefolder = "preferred/"
# make a frequency array
n = 148
dt = 11.56
freqs = np.fft.fftfreq(n,dt)
freqs = freqs[1:int(n/2)]

# run over each result file and find preferred models
if True:
    folder = "results_crisp_all/"
    paths = sorted(os.listdir(folder))
    for path in paths:
        results = pickle.load(open(os.path.join(folder,path),"rb"))

	# need a mask to tell the program which pixels to ignore (out of FOV)
        if "pre" in path:
            mask = np.load("pre_mask")
        elif "post" in path:
            mask = np.load("post_mask")
        else:
            raise ValueError("time should be 'pre' or 'post'")

        preferred = kappa_fitting.compare_models_v2(results,freqs,mask=mask,sig=0.999) # can change sig here

        savepath = os.path.join(savefolder, "pref_"+path)

        pickle.dump(preferred,(open(savepath, "wb" )))
