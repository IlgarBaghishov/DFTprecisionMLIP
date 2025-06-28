import time
import numpy as np
from scipy.linalg import lstsq
from helper_functions import load_files, train_test_split


precision_to_train_on = 6
precision_to_calc_errors_on = 6
twojmax = 6
eweight = 150

file_name_structures = "data/Be_structures.h5"
file_name_energies = "data/Be_prec_" + str(precision_to_train_on) + ".h5"
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
hlfpnt1, hlfpnt2, energy_mask, force_mask = train_test_split(df_structures["ASEatoms"])[:4]

aw = np.load('data/aw_' + str(twojmax) + '.npy')
bw_train = np.load('data/bw_' + str(precision_to_train_on) + '.npy')
bw_errors = np.load('data/bw_' + str(precision_to_calc_errors_on) + '.npy')

start_time = time.time()
aw[energy_mask] *= eweight
bw_train[energy_mask] *= eweight
coeffs, *_ = lstsq(aw[hlfpnt1], bw_train[hlfpnt1], 1.0e-13)
aw[energy_mask] /= eweight
print("Fitting finished in", time.time()-start_time, "sec")
residual = np.dot(aw,coeffs) - bw_errors
print("Energy training RMSE is", np.sqrt(np.mean(np.square(residual[hlfpnt1][energy_mask[hlfpnt1]]))))
print("Force training RMSE is", np.sqrt(np.mean(np.square(residual[hlfpnt1][force_mask[hlfpnt1]]))))
print("Energy testing RMSE is", np.sqrt(np.mean(np.square(residual[hlfpnt2][energy_mask[hlfpnt2]]))))
print("Force testing RMSE is", np.sqrt(np.mean(np.square(residual[hlfpnt2][force_mask[hlfpnt2]]))))
print("Energy RMSE if tested on the entire dataset is", np.sqrt(np.mean(np.square(residual[energy_mask]))))
print("Force RMSE if tested on the entire dataset is", np.sqrt(np.mean(np.square(residual[force_mask]))))
