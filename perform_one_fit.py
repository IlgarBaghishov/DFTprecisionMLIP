import os, time, sys
import numpy as np
from scipy.linalg import lstsq
from helper_functions import load_files, train_test_split


twojmax = int(sys.argv[1])
precision_to_train_on = int(sys.argv[2])
precision_to_calc_errors_on = int(sys.argv[3])
eweight = int(sys.argv[4])

data_dir = "data"
file_name_structures = os.path.join(data_dir, "Be_structures.h5")
file_name_energies = os.path.join(data_dir, "Be_prec_" + str(precision_to_train_on) + ".h5")
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
train_idxs, test_idxs, energy_mask, force_mask = train_test_split(
    df_structures["ASEatoms"]
)[:4]

aw = np.load(os.path.join(data_dir, "aw_" + str(twojmax) + ".npy"))
bw_train = np.load(os.path.join(
    data_dir,
    "numpy_matrices_for_fitting",
    "bw_" + str(precision_to_train_on) + ".npy"
))
bw_errors = np.load(os.path.join(
    data_dir,
    "numpy_matrices_for_fitting",
    "bw_" + str(precision_to_calc_errors_on) + ".npy"
))

start_time = time.time()
aw[energy_mask] *= eweight
bw_train[energy_mask] *= eweight
coeffs, *_ = lstsq(aw[train_idxs], bw_train[train_idxs], 1.0e-13)
aw[energy_mask] /= eweight
print("Fitting finished in", time.time() - start_time, "sec")
residual = np.dot(aw, coeffs) - bw_errors
print(
    "Energy training RMSE is",
    np.sqrt(np.mean(np.square(residual[train_idxs][energy_mask[train_idxs]]))),
)
print(
    "Force training RMSE is",
    np.sqrt(np.mean(np.square(residual[train_idxs][force_mask[train_idxs]]))),
)
print(
    "Energy testing RMSE is",
    np.sqrt(np.mean(np.square(residual[test_idxs][energy_mask[test_idxs]]))),
)
print(
    "Force testing RMSE is",
    np.sqrt(np.mean(np.square(residual[test_idxs][force_mask[test_idxs]]))),
)
print(
    "Energy RMSE if tested on the entire dataset is",
    np.sqrt(np.mean(np.square(residual[energy_mask]))),
)
print(
    "Force RMSE if tested on the entire dataset is",
    np.sqrt(np.mean(np.square(residual[force_mask]))),
)
