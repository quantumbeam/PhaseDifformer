import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pickle
from pymatgen.ext.matproj import MPRester
import sys
sys.path.append("Data/")
import utils
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from matplotlib import pyplot as plt
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.signal import find_peaks
import glob
sys.path.append("../")
from DatabaseMatching import sandman


def peak_detection(angle, pattern):
    peak_index = find_peaks(pattern)[0]
    return angle[peak_index], pattern[peak_index]


def load_data(path):
    with open(path, 'rb') as f:
        all_data = pickle.load(f)
    return all_data


def concat_dataset(paths, path_save, max_phase_num, rate):
    data_save = []
    for i in range(len(paths)):
        data = load_data(paths[i])
        use_data_num = int(len(data) * rate[i])
        print("Use data num: " + str(use_data_num))
        print(paths[i])
        use_data = data[:use_data_num]
        for j in range(len(use_data)):
            x = use_data[j].x
            intensity = use_data[j].intensity
            attr = use_data[j].attr
            components_num = use_data[j].components_num
            source_index_tmp = use_data[j].source_index
            source_index = -torch.ones(1, max_phase_num)
            source_index[0, :source_index_tmp.shape[1]] = source_index_tmp[0, :]
            weights_tmp = use_data[j].weights
            weights = -torch.ones(1, max_phase_num)
            weights[0, :weights_tmp.shape[1]] = weights_tmp[0, :]
            attr_detected_tmp = use_data[j].attr_detected
            attr_detected = torch.zeros(attr_detected_tmp.shape[0], max_phase_num)
            attr_detected[:, :attr_detected_tmp.shape[1] - 1] = attr_detected_tmp[:, :attr_detected_tmp.shape[1] - 1]
            attr_detected[:, -1] = attr_detected_tmp[:, -1]
            x_detected = use_data[j].x_detected
            intensity_detected = use_data[j].intensity_detected
            intensity_detected_raw = use_data[j].intensity_detected_raw
            mixed_pattern = use_data[j].mixed_pattern
            angle_pattern = use_data[j].angle_pattern

            dat = Data(x=torch.tensor(x), intensity=torch.tensor(intensity),
                       attr=attr, components_num=components_num, source_index=source_index, weights=weights,
                       attr_detected=attr_detected, x_detected=x_detected,
                       intensity_detected=intensity_detected,
                       intensity_detected_raw=intensity_detected_raw,
                       mixed_pattern=mixed_pattern, angle_pattern=angle_pattern)
            data_save.append(dat)

    print(data_save[0])
    print(data_save[0].attr_detected)
    print(data_save[0].weights)
    print(data_save[0].source_index)
    print(len(data_save))

    with open(path_save, mode='wb') as fo:
        pickle.dump(data_save, fo)


def make_single_data(
        components_num, all_structure, angle_min, angle_max, dim, fwhm, weight_min, weights_fixed=None):
    num_structure = len(all_structure)
    flag_list = [True for i in range(components_num)]
    while any(flag_list):
        index = np.random.randint(0, num_structure, components_num)

        structures = [all_structure[index[i]] for i in range(index.shape[0])]

        pattern_list = []
        angle_list = []
        intensity_list = []
        for i in range(len(structures)):
            angle_pattern, pattern, angle, intensity = utils.generate_xrd(structures[i], angle_min, angle_max, dim, fwhm)
            pattern_list.append(pattern)
            angle_list.append(angle)
            intensity_list.append(intensity)
            flag_list[i] = (torch.sum(torch.logical_and(angle > angle_min, angle < angle_max)) < 1)

    if weights_fixed is None:
        weights = np.random.rand(components_num)
        weights = weights / np.sum(weights)
        while np.sum(weights < weight_min) > 0:
            weights = np.random.rand(components_num)
            weights = weights / np.sum(weights)
    else:
        weights = weights_fixed

    indices_list = []
    for i in range(components_num):
        intensity_list[i] = \
            weights[i] * intensity_list[i][
                torch.logical_and(angle_list[i] > angle_min, angle_list[i] < angle_max)]
        angle_list[i] = angle_list[i][torch.logical_and(angle_list[i] > angle_min, angle_list[i] < angle_max)]
        pattern_list[i] = weights[i] * pattern_list[i]
        indices_list.append(torch.ones_like(angle_list[i]) * i)

    angle_list = torch.concat(angle_list)
    intensity_list = torch.concat(intensity_list)
    indices_list = torch.concat(indices_list)

    pattern_list.append(np.zeros_like(pattern_list[-1]))

    pattern_list = np.array(pattern_list)
    mixed_pattern = np.sum(pattern_list, axis=0)
    max_val = np.max(mixed_pattern)
    mixed_pattern = mixed_pattern / max_val
    pattern_list = pattern_list / max_val
    intensity_list = intensity_list / max_val

    angle_list, sort_info = torch.sort(angle_list)
    intensity_list = intensity_list[sort_info]
    indices_list = indices_list[sort_info]

    index_tmp = -torch.ones(components_num + 1)
    index_tmp[:components_num] = torch.tensor(index)
    index = index_tmp
    weights_tmp = -torch.ones(components_num + 1)
    weights_tmp[:components_num] = torch.tensor(weights)
    weights = weights_tmp

    angle_detected, intensity_detected_raw = peak_detection(
        angle_pattern, mixed_pattern
    )
    conv_mat = utils.generate_gaussian_convolution_mat(angle_detected, fwhm, angle_min, angle_max, dim)
    conv_mat_pinv = np.linalg.pinv(conv_mat) #[peak_num, dim]

    attr_detected = np.dot(conv_mat_pinv, pattern_list.T) #[peak_num, comp_num]
    intensity_detected = np.sum(attr_detected, axis=-1)
    intensity_detected = intensity_detected / np.max(intensity_detected)
    attr_detected = attr_detected / np.sum(attr_detected, axis=-1, keepdims=True)

    angle_pattern = torch.tensor(angle_pattern)
    mixed_pattern = torch.tensor(mixed_pattern)
    angle_detected = torch.tensor(angle_detected)
    intensity_detected = torch.tensor(intensity_detected)
    attr_detected = torch.tensor(attr_detected)
    intensity_detected_raw = torch.tensor(intensity_detected_raw)

    return angle_pattern, mixed_pattern, angle_list, intensity_list, indices_list, angle_detected, intensity_detected, attr_detected, intensity_detected_raw, index, weights, components_num


def make_data(paths, components_num, angle_min, angle_max, dim, fwhm, data_num, path_save, weight_min, weights_fixed=None):
    print("Making " + str(components_num) + " data...")
    if weights_fixed is not None:
        print("Use fixed weight: " + str(weights_fixed))
    else:
        print("not fixed weight")

    print(paths)
    all_structure = load_data(paths[0])
    for i in range(len(paths)-1):
        all_structure += load_data(paths[i+1])
    print("Structure Num: " + str(len(all_structure)))

    data_list = []
    for i in tqdm(range(data_num)):
        angle_pattern, mixed_pattern, angle_list, intensity_list, indices_list, angle_detected, intensity_detected, attr_detected, intensity_detected_raw, index, weights, components_num\
            = make_single_data(
            components_num, all_structure, angle_min, angle_max, dim, fwhm, weight_min,
            weights_fixed=weights_fixed
        )

        data = Data(
            x=torch.tensor(angle_list), intensity=torch.tensor(intensity_list),
            attr=indices_list, components_num=components_num,
            source_index=index.reshape(1, components_num+1), weights=weights.reshape(1, components_num+1),
            x_detected=angle_detected, intensity_detected=intensity_detected, attr_detected=attr_detected,
            intensity_detected_raw=intensity_detected_raw,
            mixed_pattern=mixed_pattern.reshape(1, -1), angle_pattern=angle_pattern.reshape(1, -1))

        data_list.append(data)

    print(data_list[0])
    for key in data_list[0].keys():
        print(key)
        print(data_list[0][key])

    if path_save is not None:
        with open(path_save, mode='wb') as fo:
            pickle.dump(data_list, fo)


def make_database(crystal_systems, split, save_dir):
    paths = []
    for i in range(len(crystal_systems)):
        paths.append(
            "MP_raw/" + str(crystal_systems[i]) + "/" + str(crystal_systems[i]) + "_" + str(split) + ".pkl"
        )

    data_processed = []
    os.makedirs(save_dir, exist_ok=True)
    path_save = save_dir + "/" + str(split) + ".pkl"
    for i in range(len(paths)):
        dataset = load_data(paths[i])
        angle_min = 0
        angle_max = 60
        dim = 2000
        fwhm = 0.3


        for i in tqdm(range(len(dataset))):
            data = {}
            structure = dataset[i]

            angle, pattern, angle_set, intensity_set = utils.generate_xrd(structure, angle_min, angle_max, dim, fwhm)
            angle_detected, peak_detected = peak_detection(angle, pattern)

            data["structure"] = structure
            data["angle_detected"] = angle_detected
            data["peak_detected"] = peak_detected

            data_processed.append(data)

    print("Database " + split + ": " + str(len(data_processed)))

    with open(path_save, mode='wb') as fo:
        pickle.dump(data_processed, fo)


if __name__ == "__main__":
    angle_min = 0
    angle_max = 60
    dim = 2000
    fwhm = 0.3
    weight_min = 0.1
    data_num = {
        "train": 6000,
        "val": 600,
        "test": 300
    }
    systems = ["cubic", "tetragonal", "orthorhombic"]

    print("Make training and validation data...")
    weights_fixed = None

    for split in ["train", "val"]:
        paths = []
        for i in range(len(systems)):
            paths.append("MP_raw/" + systems[i] + "/" + systems[i] + "_" + split + ".pkl")
        for components_num in range(3):
            os.makedirs("MP/Only90deg_" + str(components_num+1) + "phase", exist_ok=True)
            path_save = "MP/Only90deg_" + str(components_num+1) + "phase/" + str(split) + ".pkl"

            make_data(
                paths, components_num+1, angle_min, angle_max, dim, fwhm,
                data_num[split], path_save, weight_min, weights_fixed=weights_fixed
            )

    os.makedirs("MP/Only90deg_under3phase", exist_ok=True)
    concat_dataset(
        paths = ["MP/Only90deg_" + str(i+1) + "phase/train.pkl" for i in range(3)],
        path_save = "MP/Only90deg_under3phase/train.pkl",
        max_phase_num = 4,
        rate = [1.0, 1.0, 1.0]
    )
    os.makedirs("MP/Only90deg_under3phase", exist_ok=True)
    concat_dataset(
        paths=["MP/Only90deg_" + str(i + 1) + "phase/val.pkl" for i in range(3)],
        path_save="MP/Only90deg_under3phase/val.pkl",
        max_phase_num=4,
        rate=[1.0, 1.0, 1.0]
    )

    print("Make test data...")
    for i in range(len(systems)):
        use_systems = [systems[_] for _ in range(i + 1)]
        paths = ["MP_raw/" + use_systems[_] + "/" + use_systems[_] + "_test.pkl" for _ in range(len(use_systems))]
        for components_num in range(3):
            weights_fixed = [1 / (components_num + 1) for _ in range(components_num + 1)]
            weights_fixed = None

            file_name = use_systems[0]
            for system in use_systems[1:]:
                file_name = file_name + "_" + str(system)

            os.makedirs("MP/" + str(file_name) + "_" + str(components_num + 1) + "phase", exist_ok=True)
            path_save = "MP/" + str(file_name) + "_" + str(components_num + 1) + "phase/test.pkl"

            make_data(
                paths, components_num + 1, angle_min, angle_max, dim, fwhm,
                data_num["test"], path_save, weight_min, weights_fixed=weights_fixed
            )

    print("Make database...")
    for split in ["train", "val", "test"]:
        make_database(systems, split, save_dir="MP_database/Only90deg")

    print("Add search result...")
    splits = ["train", "val", "test"]
    for split in splits:
        paths = glob.glob("MP/*/" + split + ".pkl")
        for i in tqdm(range(len(paths))):
            sandman.add_search_result(
                paths[i], "MP_database/Only90deg/" + split + ".pkl",
                paths[i], 4475, 0.01, None
            )