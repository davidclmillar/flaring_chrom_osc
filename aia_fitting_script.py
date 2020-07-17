import os
from kappa_fitting import modelk, model1, model2, siglvl, timeseries
import numpy as np
import pickle

savefolder = "save_folder"

dt = 24

folder = "aia_cube_folder" # path to folder containing aia data cubes
paths = sorted(os.listdir(folder))         # list of filenames
for path in paths:
    data = np.load(os.path.join(folder,path)) # load a datacube (time,row,column)

    # set result arrays

    model1_parameters = np.zeros(shape=(3,data.shape[1],data.shape[2]))
    model2_parameters = np.zeros(shape=(6,data.shape[1],data.shape[2]))
    modelk_parameters = np.zeros(shape=(6,data.shape[1],data.shape[2]))
    WRS_results       = np.zeros(shape=(3,data.shape[1],data.shape[2]))

    for row in range(data.shape[1]):
        print("\r started row %d"%row,end="")		#loop over both space axes
        for col in range(data.shape[2]):

# --------------------------------------------------------------------
# fit each model, and if it fails assign parameters 1e-10 and WRS 1e10
# --------------------------------------------------------------------

            x = timeseries(data[:, row, col],dt=24.0)
# M1 fitting
            try:
                result = x.fit_M1(plot=False)
                model1_parameters[:, row, col] = result[0]
                WRS_results[0, row, col] =  result[1]
            except:
                model1_parameters[:, row, col], WRS_results[0, row, col] = np.full(3,1e-10), 1e10

# M2 fitting (two attempts)
            fail1 = 0
            try:
                result1 = x.fit_M2(plot=False)
                WRS1    = result1[1]
            except:
                fail1 = 1
            fail2 = 0
            try:
                result2 = x.fit_M2(plot=False,guess=[1e-4, 1, 1e2, -5.5, 0.1, 1],limits= [[1e-12, 0.01, 1e2, -7, 0.1, 1e-2],[1e4, 6, 1e3, -3, 0.2, np.inf]])
                WRS2 =  result2[1]
            except:
                fail2 = 1


            if fail1+fail2==2:
                model2_parameters[:, row, col], WRS_results[1, row, col] = np.full(6,1e-10), 1e10

            elif WRS2 > WRS1:
                model2_parameters[:, row, col] = result1[0]
                WRS_results[1, row, col] =  result1[1]
            else:
                model2_parameters[:, row, col] = result2[0]
                WRS_results[1, row, col] =  result2[1]

# kappa function fitting (two attempts)

            fail1 = 0
            try:
                result1 = x.fit_kappa(plot=False,guess=[2.18e+0, -1.4e-01,4.43e+02,5.0e+06,4.67e-03,1.06e-06])
                WRS1    = result1[1]
            except:
                fail1 = 1

            fail2 = 0
            try:
                result2 = x.fit_kappa(plot=False, guess=[6e-00, 0, 4.76849553e+02, 1e+0,5e-4, 3e+00])
                WRS2 =  result2[1]
            except:
                fail2 = 1


            if fail1+fail2==2:
                modelk_parameters[:, row, col], WRS_results[2, row, col] = np.full(6,1e-10), 1e10
            elif WRS2 > WRS1:
                modelk_parameters[:, row, col] = result1[0]
                WRS_results[2, row, col] =  result1[1]
            else:
                modelk_parameters[:, row, col] = result2[0]
                WRS_results[2, row, col] =  result2[1]

    all_results = [model1_parameters,model2_parameters,modelk_parameters,WRS_results]
    savepath = os.path.join(savefolder,path[:-4]+".p")
    pickle.dump(all_results,(open(savepath, "wb" )))
    del data
