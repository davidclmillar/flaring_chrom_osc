import numpy as np
import pickle
import kappa_fitting
import os

savefolder =  "savefolder"

# make a frequency array
n = 80
dt = 24.0
freqs = np.fft.fftfreq(n,dt)
freqs = freqs[1:int(n/2)]

folder = "results_folder"
paths = sorted(os.listdir(folder))
# run through folder with results and find preferred models at some significance level
for path in paths:
    results = pickle.load(open(os.path.join(folder,path),"rb"))
    dims = results[0].shape
    mask = np.ones(shape=(dims[1],dims[2]))
    preferred = kappa_fitting.compare_models_v2(results,freqs,mask=mask,sig=0.999)
    savepath = os.path.join(savefolder,dt_string+ "_pref_"+path[15:])
    pickle.dump(preferred,(open(savepath, "wb" )))
