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

    n_configs = structure_series.shape[0]
    last_index = 0
    config_to_rows_map = []
    for i in range(n_configs):
        config_to_rows_map.append(
            [last_index + j for j in range(1 + 3 * len(structure_series.values[i]))]
        )
        last_index += 1 + 3 * len(structure_series.values[i])
    config_idxs_shuffled = [i for i in range(n_configs)]
    random.seed(58)
    random.shuffle(config_idxs_shuffled)
    train_test_split_idx = int(len(config_idxs_shuffled) * (1 - test_fraction))
    train_idxs = [
        item
        for sublist_index in config_idxs_shuffled[:train_test_split_idx]
        for item in config_to_rows_map[sublist_index]
    ]
    test_idxs = [
        item
        for sublist_index in config_idxs_shuffled[train_test_split_idx:]
        for item in config_to_rows_map[sublist_index]
    ]

    energy_idxs = [i[0] for i in config_to_rows_map]
    energy_mask = np.zeros(last_index, dtype=bool)
    energy_mask[energy_idxs] = True
    force_mask = ~energy_mask

    return (
        train_idxs,
        test_idxs,
        energy_mask,
        force_mask,
        train_test_split_idx,
        config_idxs_shuffled,
        config_to_rows_map,
    )
