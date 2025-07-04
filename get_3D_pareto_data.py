import os
import sys
import numpy as np
import pandas as pd
from helper_functions import load_files, train_test_split


twojmax = int(sys.argv[1])
eweights = [5, 10, 12.25, 50, 150, 300]  # List of energy weight values
n_repetitions = int(
    sys.argv[2]
)  # How many times to repeat subsampling and training to collect statistics
subsample_size = int(
    sys.argv[3]
)  # Number of configurations to subsample according to leverage scores

data_dir = "../.."
file_name_structures = os.path.join(data_dir, "Be_structures.h5")
file_name_energies = os.path.join(data_dir, "Be_prec_6.h5")
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
(
    train_idxs,
    test_idxs,
    energy_mask,
    force_mask,
    train_test_split_idx,
    config_idxs_shuffled,
    config_to_rows_map,
) = train_test_split(df_structures["ASEatoms"])

aw = np.load(os.path.join(data_dir, "numpy_matrices_for_fitting", "aw_" + str(twojmax) + ".npy"))
bw_list = [
    np.load(os.path.join(data_dir, "numpy_matrices_for_fitting", f))
    for f in ["bw_1.npy", "bw_2.npy", "bw_3.npy", "bw_4.npy", "bw_5.npy", "bw_6.npy"]
]

df = pd.read_csv(os.path.join(data_dir, "leverage_scores_dataframe", "df_leverage.csv"), index_col=0)
probabilities = df["lev"].values / df["lev"].sum()
results = []
for j in range(n_repetitions):

    slctd = np.random.choice(df.index, subsample_size, replace=False, p=probabilities)
    train_idxs_sub = [
        item
        for slctd_ind in slctd
        for item in config_to_rows_map[config_idxs_shuffled[slctd_ind]]
    ]

    for ew in eweights:

        aw[energy_mask] *= ew
        u, s, vh = np.linalg.svd(aw[train_idxs_sub], full_matrices=False)
        aw[energy_mask] /= ew

        for i, bw in enumerate(bw_list):

            print("\n\nTraining on precision", i + 1, "data with energy weight of", ew)
            bw[energy_mask] *= ew
            coeffs = vh.T @ (np.diag(np.reciprocal(s)) @ (u.T @ bw[train_idxs_sub]))
            bw[energy_mask] /= ew

            prediction = np.dot(aw, coeffs)
            residual_self = prediction - bw
            residual_high = prediction - bw_list[-1]
            results.append(
                [
                    subsample_size,
                    ew,
                    twojmax,
                    i + 1,
                    np.sqrt(
                        np.mean(
                            np.square(
                                residual_self[train_idxs_sub][
                                    energy_mask[train_idxs_sub]
                                ]
                            )
                        )
                    ),
                    np.sqrt(
                        np.mean(
                            np.square(
                                residual_self[train_idxs_sub][
                                    force_mask[train_idxs_sub]
                                ]
                            )
                        )
                    ),
                    np.sqrt(
                        np.mean(
                            np.square(
                                residual_self[train_idxs][energy_mask[train_idxs]]
                            )
                        )
                    ),
                    np.sqrt(
                        np.mean(
                            np.square(residual_self[train_idxs][force_mask[train_idxs]])
                        )
                    ),
                    np.sqrt(
                        np.mean(
                            np.square(residual_self[test_idxs][energy_mask[test_idxs]])
                        )
                    ),
                    np.sqrt(
                        np.mean(
                            np.square(residual_self[test_idxs][force_mask[test_idxs]])
                        )
                    ),
                    np.sqrt(
                        np.mean(
                            np.square(
                                residual_high[train_idxs_sub][
                                    energy_mask[train_idxs_sub]
                                ]
                            )
                        )
                    ),
                    np.sqrt(
                        np.mean(
                            np.square(
                                residual_high[train_idxs_sub][
                                    force_mask[train_idxs_sub]
                                ]
                            )
                        )
                    ),
                    np.sqrt(
                        np.mean(
                            np.square(
                                residual_high[train_idxs][energy_mask[train_idxs]]
                            )
                        )
                    ),
                    np.sqrt(
                        np.mean(
                            np.square(residual_high[train_idxs][force_mask[train_idxs]])
                        )
                    ),
                    np.sqrt(
                        np.mean(
                            np.square(residual_high[test_idxs][energy_mask[test_idxs]])
                        )
                    ),
                    np.sqrt(
                        np.mean(
                            np.square(residual_high[test_idxs][force_mask[test_idxs]])
                        )
                    ),
                ]
            )
            print(
                "Energy subsampled RMSE with precision level used for training is",
                results[-1][-12],
            )
            print(
                "Force subsampled RMSE with precision level used for training is",
                results[-1][-11],
            )
            print(
                "Energy training RMSE with precision level used for training is",
                results[-1][-10],
            )
            print(
                "Force training RMSE with precision level used for training is",
                results[-1][-9],
            )
            print(
                "Energy testing RMSE with precision level used for training is",
                results[-1][-8],
            )
            print(
                "Force testing RMSE with precision level used for training is",
                results[-1][-7],
            )
            print(
                "Energy subsampled RMSE with highest (6th) precision level is",
                results[-1][-6],
            )
            print(
                "Force subsampled RMSE with highest (6th) precision level is",
                results[-1][-5],
            )
            print(
                "Energy training RMSE with highest (6th) precision level is",
                results[-1][-4],
            )
            print(
                "Force training RMSE with highest (6th) precision level is",
                results[-1][-3],
            )
            print(
                "Energy testing RMSE with highest (6th) precision level is",
                results[-1][-2],
            )
            print(
                "Force testing RMSE with highest (6th) precision level is",
                results[-1][-1],
            )

df_results = pd.DataFrame(
    results,
    columns=[
        "Subsample size",
        "Energy Weight",
        "2Jmax",
        "Train Precision",
        "Subsampled Energy RMSE (train precision)",
        "Subsampled Force RMSE (train precision)",
        "Training Energy RMSE (train precision)",
        "Training Force RMSE (train precision)",
        "Testing Energy RMSE (train precision)",
        "Testing Force RMSE (train precision)",
        "Subsampled Energy RMSE (6th precision)",
        "Subsampled Force RMSE (6th precision)",
        "Training Energy RMSE (6th precision)",
        "Training Force RMSE (6th precision)",
        "Testing Energy RMSE (6th precision)",
        "Testing Force RMSE (6th precision)",
    ],
)
df_results.to_csv("results.csv")
