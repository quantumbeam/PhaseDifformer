import copy
import pickle
import torch
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
import pandas as pd


def load_database(path, threshold):
    print("Loading data...")
    dataset = load_data(path)

    peak_list = []
    intensity_list = []
    for i in tqdm(range(len(dataset))):
        intensity = np.array(dataset[i]["peak_detected"])
        peak_list.append(np.array(dataset[i]["angle_detected"])[intensity > threshold])
        intensity_list.append(np.array(dataset[i]["peak_detected"])[intensity > threshold])

    return peak_list, intensity_list

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def generate_gaussian_convolution_mat(angle, fwhm, angle_min, angle_max, dim):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    angle_all = np.linspace(angle_min, angle_max, dim)

    gaussian_convolution_mat = np.array(
        [gaussian(angle_all, np.array(angle[i]), sigma) for i in range(angle.shape[0])]).T

    return np.array(torch.tensor(gaussian_convolution_mat))

def load_data(path):
    with open(path, 'rb') as f:
        all_data = pickle.load(f)
    return all_data

def convolution(pos, intensity, angle_min, angle_max, dim, fwhm):
    x_vals = np.linspace(angle_min, angle_max, dim)
    y_vals = np.zeros_like(x_vals)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    for p, i in zip(pos, intensity):
        y_vals += i * gaussian(x_vals, p, sigma)
    y_vals = y_vals / np.max(y_vals)

    return x_vals, y_vals

def gaussian_convolute(xrd_pattern, angle_min, angle_max, dim, fwhm):
    x_vals = np.linspace(angle_min, angle_max, dim)
    y_vals = np.zeros_like(x_vals)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    for peak_pos, intensity in zip(xrd_pattern.x, xrd_pattern.y):
        y_vals += intensity * gaussian(x_vals, peak_pos, sigma)
    y_vals = y_vals/np.max(y_vals)

    return x_vals, y_vals, torch.tensor(xrd_pattern.x), torch.tensor(xrd_pattern.y)/100


def generate_xrd(structure, angle_min, angle_max, dim, fwhm):
    xrd_calculator = XRDCalculator(wavelength="CuKa")
    xrd_pattern = xrd_calculator.get_pattern(structure)

    return gaussian_convolute(xrd_pattern, angle_min, angle_max, dim, fwhm)


if __name__ == "__main__":
    path_dir = "/Users/yuseiito/Desktop/project/XRDDecomposition/Paper_XRDDecomposition/experimental_result/YI01_rutile+anatase_2deg_per_min"

    path_cifs = [
        "/Users/yuseiito/Desktop/project/XRDDecomposition/Paper_XRDDecomposition/experimental_result/cifs/TiO2_rutile_refined_ver1.cif",
        "/Users/yuseiito/Desktop/project/XRDDecomposition/Paper_XRDDecomposition/experimental_result/cifs/TiO2_anatase_refined_ver1.cif",
    ]

    path_save = "decompose_result.csv"

    fwhm = 0.3
    angle_min = 0
    angle_max = 60
    dim = 2000

    angle_x = np.linspace(angle_min, angle_max, dim)

    active_index = load_data(path_dir + "/active_index.pkl")[0]
    angle = torch.rad2deg(load_data(path_dir + "/angle.pkl")[0][0][0])
    dif_generate = load_data(path_dir + "/dif_generate.pkl")[0]
    generate = load_data(path_dir + "/generate.pkl")[0]
    mask = load_data(path_dir + "/mask.pkl")[0]
    mixed_intensity = load_data(path_dir + "/mixed_intensity.pkl")[0]

    try_num = 0
    for i in range(len(active_index)):
        if active_index[i][0]:
            try_num += 1

    mask_this_index = []
    dif_generate_this_index = []
    generate_this_index = []
    mixed_intensity_this_index = []
    for i in range(try_num):
        mask_this_index.append(mask[i][0])
        dif_generate_this_index.append(dif_generate[i][0])
        generate_this_index.append(generate[i][0])
        mixed_intensity_this_index.append(mixed_intensity[i][0])

    pattern_generate_list = []
    pattern_dif_generate_list = []
    for i in range(try_num):
        conv_mat = generate_gaussian_convolution_mat(
            angle[torch.logical_not(mask_this_index[i])],
            fwhm, angle_min, angle_max, dim)
        pattern_dif_generate_list.append(
            conv_mat @ np.array((dif_generate_this_index[i][torch.logical_not(mask_this_index[i])]))
        )
        pattern_generate_list.append(
            conv_mat @ np.array((generate_this_index[i][torch.logical_not(mask_this_index[i])]))
        )

    structures = [Structure.from_file(path_cifs[i]) for i in range(len(path_cifs))]
    patterns = []
    mixed_pattern = []
    for i in tqdm(range(len(structures))):
        structure = structures[i]
        angle, pattern, angle_set, intensity_set = generate_xrd(structure, angle_min, angle_max, dim, fwhm)
        patterns.append(pattern)

    df = pd.DataFrame()
    df["angle"] = np.array(angle_x)
    plt.figure()
    for i in range(try_num-1):
        plt.plot(
            angle_x, pattern_dif_generate_list[i]/(np.max(pattern_dif_generate_list[i])+0.00001),
            alpha=0.6, label="decomposed " + str(i+1)
        )
        df["decomposed" + str(i+1)] = np.array(
            pattern_dif_generate_list[i]/(np.max(pattern_dif_generate_list[i])+0.00001))

    i = try_num-1
    plt.plot(
        angle_x, pattern_generate_list[i] / (np.max(pattern_generate_list[i]) + 0.00001),
        label="decomposed " + str(i+1), alpha=0.6
    )
    df["decomposed" + str(i + 1)] = np.array(
        pattern_dif_generate_list[i] / (np.max(pattern_dif_generate_list[i]) + 0.00001))

    for i in range(len(patterns)):
        plt.plot(
            angle_x, patterns[i] / (np.max(patterns[i]) + 0.00001),
            label="reference " + str(i + 1), alpha=0.6
        )
        df["reference" + str(i+1)] = np.array(patterns[i] / (np.max(patterns[i]) + 0.00001))
    plt.legend()
    plt.xlim(10, 60)

    df.to_csv(path_save)
    plt.show()