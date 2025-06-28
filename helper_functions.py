import pandas as pd
import numpy as np
import random


def load_files(file_name_structures, file_name_energies):
    df_structures = pd.read_hdf(file_name_structures)
    df_structures.sort_index(inplace=True)

    df_energies = pd.read_hdf(file_name_energies)
    df_energies.sort_values(by=["index"], inplace=True)

    df_structures = df_structures[df_structures.index.isin(df_energies["index"].values)]
    return df_structures, df_energies


def train_test_split(structure_series, test_fraction=0.5):

    configs_num = structure_series.shape[0]
    last_index = 0
    configs_index = []
    for i in range(configs_num):
        configs_index.append([last_index+j for j in range(1+3*len(structure_series.values[i]))])
        last_index += 1+3*len(structure_series.values[i])
    ind_configs_index = [i for i in range(configs_num)]
    random.seed(58)
    random.shuffle(ind_configs_index)
    hlfpnt = int(len(ind_configs_index) * (1-test_fraction))
    hlfpnt1 = [item for sublist_index in ind_configs_index[:hlfpnt] for item in configs_index[sublist_index]]
    hlfpnt2 = [item for sublist_index in ind_configs_index[hlfpnt:] for item in configs_index[sublist_index]]

    energy_idxs = [i[0] for i in configs_index]
    energy_mask = np.zeros(last_index, dtype=bool)
    energy_mask[energy_idxs] = True
    force_mask = ~energy_mask

    return hlfpnt1, hlfpnt2, energy_mask, force_mask, hlfpnt, ind_configs_index, configs_index
    # return train_idxs, test_idxs