import sys
import numpy as np
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.scrapers.ase_funcs import get_apre, ase_scraper
from helper_functions import load_files


def ase_scraper(snap, frames, energies, forces, stresses):
    """
    Function to organize groups and allocate shared arrays used in Calculator. For now when using
    ASE frames, we don't have groups.

    Args:
        s: fitsnap instance.
        data: List of ASE frames or dictionary group table containing frames.

    Returns a list of data dictionaries suitable for fitsnap descriptor calculator.
    If running in parallel, this list will be distributed over procs, so that each proc will have a
    portion of the list.
    """

    snap.data = [
        collate_data(snap, indx, len(frames), a, e, f, s)
        for indx, (a, e, f, s) in enumerate(zip(frames, energies, forces, stresses))
    ]


def collate_data(s, indx, size, atoms, energy, forces, stresses):
    """
    Function to organize fitting data for FitSNAP from ASE atoms objects.

    Args:
        atoms: ASE atoms object for a single configuration of atoms.
        name: Optional name of this configuration.
        group_dict: Optional dictionary containing group information.

    Returns a data dictionary for a single configuration.
    """

    # Transform ASE cell to be appropriate for LAMMPS.
    apre = get_apre(cell=atoms.cell)
    R = np.dot(np.linalg.inv(atoms.cell), apre)
    positions = np.matmul(atoms.get_positions(), R)
    cell = apre.T

    # Make a data dictionary for this config.

    data = {}
    data["PositionsStyle"] = "angstrom"
    data["AtomTypeStyle"] = "chemicalsymbol"
    data["StressStyle"] = "bar"
    data["LatticeStyle"] = "angstrom"
    data["EnergyStyle"] = "electronvolt"
    data["ForcesStyle"] = "electronvoltperangstrom"
    data["Group"] = "All"
    data["File"] = None
    data["Stress"] = stresses
    data["Positions"] = positions
    data["Energy"] = energy
    data["AtomTypes"] = atoms.get_chemical_symbols()
    data["NumAtoms"] = len(atoms)
    data["Forces"] = forces
    data["QMLattice"] = cell
    data["test_bool"] = (
        indx >= s.config.sections["GROUPS"].group_table["All"]["training_size"] * size
    )
    data["Lattice"] = cell
    data["Rotation"] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    data["Translation"] = np.zeros((len(atoms), 3))
    data["eweight"] = s.config.sections["GROUPS"].group_table["All"]["eweight"]
    data["fweight"] = s.config.sections["GROUPS"].group_table["All"]["fweight"]
    data["vweight"] = s.config.sections["GROUPS"].group_table["All"]["vweight"]

    return data


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

twojmax = int(sys.argv[1])
precision = int(sys.argv[2])

settings = {
    "BISPECTRUM": {
        "numTypes": 1,
        "twojmax": twojmax,
        "rcutfac": 4.812302818,
        "rfac0": 0.99363,
        "rmin0": 0.0,
        "wj": 1.0,
        "radelem": 0.5,
        "type": "Be",
        "wselfallflag": 0,
        "chemflag": 0,
        "bzeroflag": 1,
        "quadraticflag": 1,
    },
    "CALCULATOR": {
        "calculator": "LAMMPSSNAP",
        "energy": 1,
        "force": 1,
        "stress": 0,
    },
    "ESHIFT": {"Be": 0.0},
    "GROUPS": {
        # name size eweight fweight vweight
        "group_sections": "name training_size testing_size eweight fweight vweight",
        "group_types": "str float float float float float",
        "smartweights": 0,
        "random_sampling": 0,
        "All": "0.5    0.5    1.0    1.0    0.0",
    },
    "OUTFILE": {"metrics": "Be_metrics.md", "potential": "Be_pot"},
    "REFERENCE": {
        "units": "metal",
        "atom_style": "atomic",
        "pair_style": "zero 10.0",
        "pair_coeff": "* *",
    },
    "SOLVER": {"solver": "SVD", "compute_testerrs": 1, "detailed_errors": 1},
    "EXTRAS": {
        "dump_descriptors": 0,
        "dump_truth": 0,
        "dump_weights": 0,
        "dump_dataframe": 0,
    },
    "MEMORY": {"override": 0},
}

file_name_structures = "../Be_structures.h5"
file_name_energies = "../Be_prec_" + str(precision) + ".h5"
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
configs_num = df_structures["ASEatoms"].shape[0]
ratio = configs_num // size
rem = configs_num % size
a1 = rank * ratio + min(rank, rem)
a2 = (rank + 1) * ratio + min(rank, rem - 1) + 1
fs_instance = FitSnap(settings, comm=comm, arglist=["--overwrite"])
ase_scraper(
    fs_instance,
    df_structures["ASEatoms"].values[a1:a2],
    df_energies["energy"].values[a1:a2],
    df_energies["forces"].values[a1:a2],
    df_energies["stress"].values[a1:a2],
)
fs_instance.process_configs(allgather=True)


if rank == 0:

    a = fs_instance.pt.shared_arrays["a"].array
    b = fs_instance.pt.shared_arrays["b"].array

    np.save("aw_" + str(twojmax) + ".npy", a)
    np.save("bw_" + str(precision) + ".npy", b)
