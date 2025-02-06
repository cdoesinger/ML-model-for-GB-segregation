import numpy as np

import pandas as pd

import sklearn.gaussian_process
import sklearn.metrics, sklearn.model_selection

import regression_functions 

descriptors_structure = "R15" # "Q10" # 
descriptors_solute    = ["E_ref_0", "E_ref_6", 
                         "E_ref_9", "E_ref_10", 
                         "E_coh_MP", "Vol_perAtom_MatMiner"]


### read data
print("read data: start --- ", end="", flush=True)
df     = pd.read_csv("segregation_data.csv")
y_data = df.loc[:, "E_SEG"].to_numpy()
solute = df.loc[:, "SOLUTE"].to_numpy()
XYZ    = df.loc[:, "X001":"Z120"].to_numpy()

df_el_data = pd.read_csv("element_data.csv", index_col="Element")
print("finish", flush=True)


### calculate structural descriptors
print("calculate descriptors: start --- ", end="", flush=True)
if descriptors_structure == "R15":
    x_data = regression_functions.calculate_rad_dist(XYZ=XYZ, n_neighbours=15)
elif descriptors_structure == "Q10":
    x_data = regression_functions.calculate_Steinhardt_Par(XYZ=XYZ, l=[2,3,4,5,6,7,8,9,10], r_cut=3.5)
else:
    """something else"""
    x_data = np.random.randn(XYZ.shape[0], 5)
print("finish part1 --- ", end="", flush=True)


### add descriptors for the solute element
for descr_el in descriptors_solute:
    pars = df_el_data.loc[solute, descr_el].to_numpy().reshape((-1,1))
    x_data = np.append(x_data, pars, axis=1)
print("finish part2", flush=True)


### select valid data, i.e. remove datapoints without E_SEG
# "segregation_data.csv" containes all sites in the structure, including sites without calculations
x_train      = x_data[~np.isnan(y_data),:]
y_train      = y_data[~np.isnan(y_data)]
solute_train = solute[~np.isnan(y_data)]


### Do a cross-validation
n_folds  =  5
n_repeat =  1
R2    = np.zeros(n_folds*n_repeat)
RMSE  = np.zeros(n_folds*n_repeat)

rep_cv = sklearn.model_selection.RepeatedKFold(n_splits=n_folds, n_repeats=n_repeat)
for icv, (itrain, itest) in enumerate(rep_cv.split(x_train)):
    print(f"CV fold {icv+1:4d} of {n_folds*n_repeat:4d}")

    ## train the GPR
    gpr       = regression_functions.train_GPR(x_train[itrain,:], y_train[itrain])

    ## do predictions
    pred_test = gpr.predict(x_train[itest,:])

    ## evaluate scores
    R2[icv]   =  sklearn.metrics.r2_score(y_train[itest], pred_test)
    RMSE[icv] =  sklearn.metrics.mean_squared_error(y_train[itest], pred_test, squared=False)

print(f"{'R2-score:':10s}{R2.mean():8.3f} +/- {R2.std():8.3f}")
print(f"{'RMSE:':10s}{RMSE.mean():8.3f} +/- {RMSE.std():8.3f}")