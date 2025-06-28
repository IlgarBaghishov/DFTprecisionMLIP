import time
import numpy as np
import pandas as pd
from helper_functions import load_files, train_test_split


twojmax = 6
eweight = 150

file_name_structures = "../Be_structures.h5"
file_name_energies = "../Be_prec_6.h5"
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
hlfpnt1, hlfpnt2, energy_mask, force_mask, hlfpnt, ind_configs_index = train_test_split(df_structures["ASEatoms"])[:6]

aw_train = np.load("../numpy_matrices_for_fitting/aw_" + str(twojmax) + ".npy")[hlfpnt1]

start_time = time.time()
aw_train[energy_mask[hlfpnt1]] *= eweight
u = np.linalg.svd(aw_train, full_matrices=False)[0]

df = pd.DataFrame(np.zeros((hlfpnt,2)),columns=['lev','block_lev'])
diag_elms = np.sum(u**2,axis=1)
i_strt = 0
for i,j in enumerate(ind_configs_index[:hlfpnt]):
    n_atms = len(df_structures["ASEatoms"].values[j])
    df.loc[i,'lev'] = diag_elms[i_strt]
    df.loc[i,'block_lev'] = np.sum(diag_elms[i_strt:(i_strt+1+3*n_atms)])
    i_strt += 1 + 3*n_atms
    print(i, df.loc[i,'lev'], df.loc[i,'block_lev'])

print(i_strt)
df.to_csv('df_leverage.csv')
