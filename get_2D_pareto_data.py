import sys
import numpy as np
import pandas as pd
from helper_functions import load_files, train_test_split


twojmax = int(sys.argv[1])
eweights = [
    0.01,
    0.1,
    1,
    5,
    10,
    12.25,
    20,
    50,
    100,
    150,
    200,
    300,
    500,
    750,
    1000,
    1500,
    2000,
]  # List of energy weight values

file_name_structures = "../Be_structures.h5"
file_name_energies = "../Be_prec_6.h5"
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
train_idxs, test_idxs, energy_mask, force_mask = train_test_split(
    df_structures["ASEatoms"]
)[:4]

aw = np.load("../numpy_matrices_for_fitting/aw_" + str(twojmax) + ".npy")
bw_1 = np.load("../numpy_matrices_for_fitting/bw_1.npy")
bw_2 = np.load("../numpy_matrices_for_fitting/bw_2.npy")
bw_3 = np.load("../numpy_matrices_for_fitting/bw_3.npy")
bw_4 = np.load("../numpy_matrices_for_fitting/bw_4.npy")
bw_5 = np.load("../numpy_matrices_for_fitting/bw_5.npy")
bw_6 = np.load("../numpy_matrices_for_fitting/bw_6.npy")
bw_list = [bw_1, bw_2, bw_3, bw_4, bw_5, bw_6]

results = []
for ew in eweights:

    aw[energy_mask] *= ew
    u, s, vh = np.linalg.svd(aw[train_idxs], full_matrices=False)
    aw[energy_mask] /= ew

    for i, bw in enumerate(bw_list):

        print("\n\nTraining on precision", i + 1, "data with energy weight of", ew)
        bw[energy_mask] *= ew
        coeffs = vh.T @ (np.diag(np.reciprocal(s)) @ (u.T @ bw[train_idxs]))
        bw[energy_mask] /= ew

        prediction = np.dot(aw, coeffs)
        residual_self = prediction - bw
        residual_high = prediction - bw_6
        results.append(
            [
                ew,
                twojmax,
                i + 1,
                np.sqrt(
                    np.mean(
                        np.square(residual_self[train_idxs][energy_mask[train_idxs]])
                    )
                ),
                np.sqrt(
                    np.mean(
                        np.square(residual_self[train_idxs][force_mask[train_idxs]])
                    )
                ),
                np.sqrt(
                    np.mean(np.square(residual_self[test_idxs][energy_mask[test_idxs]]))
                ),
                np.sqrt(
                    np.mean(np.square(residual_self[test_idxs][force_mask[test_idxs]]))
                ),
                np.sqrt(
                    np.mean(
                        np.square(residual_high[train_idxs][energy_mask[train_idxs]])
                    )
                ),
                np.sqrt(
                    np.mean(
                        np.square(residual_high[train_idxs][force_mask[train_idxs]])
                    )
                ),
                np.sqrt(
                    np.mean(np.square(residual_high[test_idxs][energy_mask[test_idxs]]))
                ),
                np.sqrt(
                    np.mean(np.square(residual_high[test_idxs][force_mask[test_idxs]]))
                ),
            ]
        )
        print(
            "Energy training RMSE with precision level used for training is",
            results[-1][-8],
        )
        print(
            "Force training RMSE with precision level used for training is",
            results[-1][-7],
        )
        print(
            "Energy testing RMSE with precision level used for training is",
            results[-1][-6],
        )
        print(
            "Force testing RMSE with precision level used for training is",
            results[-1][-5],
        )
        print(
            "Energy training RMSE with highest (6th) precision level is",
            results[-1][-4],
        )
        print(
            "Force training RMSE with highest (6th) precision level is", results[-1][-3]
        )
        print(
            "Energy testing RMSE with highest (6th) precision level is", results[-1][-2]
        )
        print(
            "Force testing RMSE with highest (6th) precision level is", results[-1][-1]
        )

df_results = pd.DataFrame(
    results,
    columns=[
        "Energy Weight",
        "2Jmax",
        "Train Precision",
        "Training Energy RMSE (train precision)",
        "Training Force RMSE (train precision)",
        "Testing Energy RMSE (train precision)",
        "Testing Force RMSE (train precision)",
        "Training Energy RMSE (6th precision)",
        "Training Force RMSE (6th precision)",
        "Testing Energy RMSE (6th precision)",
        "Testing Force RMSE (6th precision)",
    ],
)
df_results.to_csv("results.csv")
