import pickle
from pymatgen.ext.matproj import MPRester
import random
import numpy as np
import os
from tqdm import tqdm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def get_crystal_system(structure):
    analyzer = SpacegroupAnalyzer(structure)
    crystal_system = analyzer.get_crystal_system()
    return crystal_system


def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


if __name__ == "__main__":
    API_KEY = "sPEcL73JNS3dypIB7nqwdRQ25Npg9Duq"
    len_min = 2
    len_max = 10
    site_max = 200

    os.makedirs(f"MP_raw", exist_ok=True)

    mpr = MPRester(API_KEY)

    all_structures = []

    print("Download structure data...")
    docs = []
    for i in tqdm(range(site_max)):
        criteria = {"nsites": i+1}
        with MPRester(API_KEY) as mpr:
            docs += mpr.get_summary(criteria)

    structure_list = []
    for i in tqdm(range(len(docs))):
        if docs[i]["structure"].lattice.a > len_min and docs[i]["structure"].lattice.a < len_max:
            if docs[i]["structure"].lattice.b > len_min and docs[i]["structure"].lattice.b < len_max:
                if docs[i]["structure"].lattice.c > len_min and docs[i]["structure"].lattice.c < len_max:
                    if docs[i]["structure"].cart_coords.shape[0] < site_max:
                        structure_list.append(docs[i]["structure"])

    print(len(structure_list))

    structures = {
        "triclinic": [],
        "monoclinic": [],
        "orthorhombic": [],
        "tetragonal": [],
        "cubic": [],
        "trigonal": [],
        "hexagonal": []
    }

    for i in tqdm(range(len(structure_list))):
        crystal_system = get_crystal_system(structure_list[i])
        structures[crystal_system].append(structure_list[i])

    with open("MP_raw/structures_len_2_10_site_200.pkl", "wb") as f:
        pickle.dump(structure_list, f)

    crystal_systems = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "cubic", "trigonal", "hexagonal"]

    for crystal_system in crystal_systems:
        print(f"Processing {crystal_system}...")

        os.makedirs(f"MP_raw/{crystal_system}", exist_ok=True)

        id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(structures[crystal_system]),
            split_seed=123,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            keep_data_order=False,
        )

        dataset_train = [structures[crystal_system][x] for x in id_train]
        dataset_val = [structures[crystal_system][x] for x in id_val]
        dataset_test = [structures[crystal_system][x] for x in id_test]

        print(crystal_system + " train: " + str(len(dataset_train)))
        print(crystal_system + " val: " + str(len(dataset_val)))
        print(crystal_system + " test: " + str(len(dataset_test)))

        with open(f"MP_raw/{crystal_system}/{crystal_system}_train.pkl", "wb") as f:
            pickle.dump(dataset_train, f)
        with open(f"MP_raw/{crystal_system}/{crystal_system}_val.pkl", "wb") as f:
            pickle.dump(dataset_val, f)
        with open(f"MP_raw/{crystal_system}/{crystal_system}_test.pkl", "wb") as f:
            pickle.dump(dataset_test, f)

        print(f"Finished processing {crystal_system}.")

    print("Dataset splitting and saving for all crystal systems completed.")