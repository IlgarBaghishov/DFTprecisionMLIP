import time, sys
import numpy as np
from scipy.linalg import lstsq
import pandas as pd
from helper_functions import load_files, train_test_split


twojmax = int(sys.argv[1])
precision_to_train_on = 6
precision_to_calc_errors_on = 6
eweight = 150
n_repetitions = 50  # How many times to repeat subsampling and training to collect statistics
subsample_size = int(sys.argv[2])  # Number of configurations to subsample

file_name_structures = "../../Be_structures.h5"
file_name_energies = "../../Be_prec_" + str(precision_to_train_on) + ".h5"
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
train_idxs, test_idxs, energy_mask, force_mask, train_test_split_idx, config_idxs_shuffled, config_to_rows_map = \
    train_test_split(df_structures["ASEatoms"])

aw = np.load('../../numpy_matrices_for_fitting/aw_' + str(twojmax) + '.npy')
bw_train = np.load('../../numpy_matrices_for_fitting/bw_' + str(precision_to_train_on) + '.npy')
bw_errors = np.load('../../numpy_matrices_for_fitting/bw_' + str(precision_to_calc_errors_on) + '.npy')
aw[energy_mask] *= eweight
bw_train[energy_mask] *= eweight
bw_errors[energy_mask] *= eweight

df = pd.read_csv("../../leverage_scores_dataframe/df_leverage.csv", index_col=0)
lev_probabilities = df["lev"].values / df["lev"].sum()
block_lev_probabilities = df["block_lev"].values / df["block_lev"].sum()
method = ['Random', 'Leverage', 'Block Leverage']
probabilities = [None, lev_probabilities, block_lev_probabilities]
results = []

for i in range(3):

    if i == 0:
        print("\n\n\nRandom Subsampling results")
    elif i == 1:
        print("\n\n\nLeverage Subsampling results")
    else:
        print("\n\n\nBlock Leverage Subsampling results")

    for j in range(n_repetitions):

        slctd = np.random.choice(df.index, subsample_size, replace=False, p=probabilities[i])
        train_idxs_sub = [item for slctd_ind in slctd for item in config_to_rows_map[config_idxs_shuffled[slctd_ind]]]

        print("\nFitting to subsampled ", subsample_size, "configurations, repetition", j)
        start_time = time.time()
        coeffs, *_ = lstsq(aw[train_idxs_sub], bw_train[train_idxs_sub], 1.0e-13)
        print("Fitting finished in", time.time()-start_time, "sec")
        residual = np.dot(aw,coeffs) - bw_errors
        results.append([
            subsample_size, eweight, twojmax, precision_to_train_on, precision_to_calc_errors_on, method[i],
            np.sqrt(np.mean(np.square(residual[train_idxs_sub][energy_mask[train_idxs_sub]])))/eweight,
            np.sqrt(np.mean(np.square(residual[train_idxs_sub][force_mask[train_idxs_sub]]))),
            np.sqrt(np.mean(np.square(residual[train_idxs][energy_mask[train_idxs]])))/eweight,
            np.sqrt(np.mean(np.square(residual[train_idxs][force_mask[train_idxs]]))),
            np.sqrt(np.mean(np.square(residual[test_idxs][energy_mask[test_idxs]])))/eweight,
            np.sqrt(np.mean(np.square(residual[test_idxs][force_mask[test_idxs]])))
        ])
        print("Energy subsampled RMSE is", results[-1][-6])
        print("Force subsampled RMSE is", results[-1][-5])
        print("Energy training RMSE is", results[-1][-4])
        print("Force training RMSE is", results[-1][-3])
        print("Energy testing RMSE is", results[-1][-2])
        print("Force testing RMSE is", results[-1][-1])

df_results = pd.DataFrame(results, columns=[
    'Subsample size', 'Energy Weight', '2Jmax', 'Train Precision', 'Error Precision', 'Sampling Method',
    'Subsampled Energy RMSE', 'Subsampled Force RMSE',
    'Training Energy RMSE', 'Training Force RMSE',
    'Testing Energy RMSE', 'Testing Force RMSE'])
df_results.to_csv("results.csv")