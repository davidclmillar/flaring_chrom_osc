from kappa_fitting import modelk, model1, model2, siglvl, timeseries
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time as timer
import pickle
import os
savefolder = "results/"
if not os.path.exists(savefolder):
    os.makedirs(savefolder)


# run over some wavelengths
for line in ["Halpha", "ca8542"]:
    if line == "Halpha":
        wls = [3,5,7,9,11,13]
    elif line == "ca8542":
        wls = [1,3,5,7,9,11,13,15,17,19,21,23]
    folder = "rebinned_by_10/%s/"%line

    for wl in wls:
        for time in ['pre','post']:
            print("\n  ")
            path = "%s_%s_%s.npy"%(line,str(wl).zfill(2),time)
            data = np.load(os.path.join(folder,path)) # load a datacube (time,row,column)

    # load relevant mask
            if "pre" in path:
                mask = np.load("pre_mask")
            elif "post" in path:
                mask = np.load("post_mask")
            else:
                raise ValueError("time should be 'pre' or 'post'")


    # set result arrays

            model1_parameters = np.zeros(shape=(3,data.shape[1],data.shape[2]))
            model2_parameters = np.zeros(shape=(6,data.shape[1],data.shape[2]))
            modelk_parameters = np.zeros(shape=(6,data.shape[1],data.shape[2]))

            WRS_results       = np.zeros(shape=(3,data.shape[1],data.shape[2]))

            start_whole = timer.time()
            for row in range(data.shape[1]):		#loop over both space axes
                for col in range(data.shape[2]):
                    print("\r row, col : %d  , %d  "%(row,col),end="")
                    if mask[row, col]==1:
                        x = timeseries(data[:, row, col])   # ignore if it is masked

    # --------------------------------------------------------------------
    # fit each model, and if it fails assign parameters 1e-10 and WRS 1e10
    # --------------------------------------------------------------------
                        try:
                            result = x.fit_M1(plot=False)
                            model1_parameters[:, row, col] = result[0]
                            WRS_results[0, row, col] =  result[1]
                        except:
                            model1_parameters[:, row, col], WRS_results[0, row, col] = np.full(3,1e-10), 1e10
                        try:
                            result = x.fit_M2(plot=False)
                            model2_parameters[:, row, col] = result[0]
                            WRS_results[1, row, col] =  result[1]
                        except:
                            model2_parameters[:, row, col], WRS_results[1, row, col] = np.full(6,1e-10), 1e10

                        try:
                            result = x.fit_kappa(plot=False)
                            modelk_parameters[:, row, col] = result[0]
                            WRS_results[2, row, col] =  result[1]
                        except:
                            modelk_parameters[:, row, col], WRS_results[2, row, col] = np.full(6,1e-10), 1e10

                    else:
                        model1_parameters[:, row, col], WRS_results[0, row, col] = np.full(3,1e-10), 1e10
                        model2_parameters[:, row, col], WRS_results[1, row, col] = np.full(6,1e-10), 1e10
                        modelk_parameters[:, row, col], WRS_results[2, row, col] = np.full(6,1e-10), 1e10

            end_whole = timer.time()
            print("\r %s: %s seconds"%(path,str(end_whole - start_whole)))


            all_results = [model1_parameters,model2_parameters,modelk_parameters,WRS_results]

            savepath = os.path.join(savefolder,savefolder[:-1]+path[:-4]+".p")

            pickle.dump(all_results,(open(savepath, "wb" )))


            del data

