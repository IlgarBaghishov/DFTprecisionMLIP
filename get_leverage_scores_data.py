import os, time, sys
import numpy as np
import pandas as pd
from helper_functions import load_files, train_test_split


twojmax = int(sys.argv[1])
eweight = 150

data_dir = ".."
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
) = train_test_split(df_structures["ASEatoms"])[:6]

aw_train = np.load(os.path.join(
    data_dir,
    "numpy_matrices_for_fitting",
    "aw_" + str(twojmax) + ".npy"
))[train_idxs]

start_time = time.time()
aw_train[energy_mask[train_idxs]] *= eweight
u = np.linalg.svd(aw_train, full_matrices=False)[0]

df = pd.DataFrame(np.zeros((train_test_split_idx, 2)), columns=["lev", "block_lev"])
diag_elms = np.sum(u**2, axis=1)
i_strt = 0
for i, j in enumerate(config_idxs_shuffled[:train_test_split_idx]):
    n_atms = len(df_structures["ASEatoms"].values[j])
    df.loc[i, "lev"] = diag_elms[i_strt]
    df.loc[i, "block_lev"] = np.sum(diag_elms[i_strt : (i_strt + 1 + 3 * n_atms)])
    i_strt += 1 + 3 * n_atms
    print(i, df.loc[i, "lev"], df.loc[i, "block_lev"])

print(i_strt)
df.to_csv("df_leverage.csv")
