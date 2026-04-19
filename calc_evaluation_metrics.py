import torch
import pickle
from scipy.stats import wasserstein_distance
import numpy as np
import json
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm
import copy

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def calc_distance(peak_list, intensity_list, target_peak, target_intensity, metric="emd"):
    if metric == "l2":
        peak_list = np.array(peak_list)
        intensity_list = np.array(intensity_list)
        target_intensity = np.array(target_intensity)
        target_peak = np.array(target_peak)
        conv_mat_peak = generate_gaussian_convolution_mat(peak_list, 0.3, 0, 60, 2000)
        target_conv_mat = generate_gaussian_convolution_mat(target_peak, 0.3, 0, 60, 2000)

        pattern = conv_mat_peak @ intensity_list
        target_pattern = target_conv_mat @ target_intensity

        pattern = pattern / (np.max(pattern)+0.00001)
        target_pattern = target_pattern / np.max(target_pattern + 0.00001)

        return np.linalg.norm(pattern-target_pattern), pattern, target_pattern

    elif metric == "emd":
        return wasserstein_distance(peak_list, target_peak, u_weights=intensity_list, v_weights=target_intensity)
    else:
        NotImplementedError()

def calc_eval_func(distance_rate):
    distance_rate = np.concatenate(distance_rate)
    num = distance_rate.shape[0]

    return distance_rate, num

def generate_gaussian_convolution_mat(angle, fwhm, angle_min, angle_max, dim):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    angle_all = np.linspace(angle_min, angle_max, dim)

    gaussian_convolution_mat = np.array(
        [gaussian(angle_all, np.array(angle[i]), sigma) for i in range(angle.shape[0])]).T

    return np.array(torch.tensor(gaussian_convolution_mat))

def eval_decompose_accuracy(path_dir, retry = False):
    active_index = load_data(path_dir + "/active_index.pkl")
    angle = load_data(path_dir + "/angle.pkl")
    attr = load_data(path_dir + "/attr.pkl")
    dif_generate = load_data(path_dir + "/dif_generate.pkl")
    generate = load_data(path_dir + "/generate.pkl")
    mask = load_data(path_dir + "/mask.pkl")
    mixed_intensity = load_data(path_dir + "/mixed_intensity.pkl")
    source_index = load_data(path_dir + "/source_index.pkl")

    batch_num = len(source_index)

    distance_rate = []
    try_num_list = []
    phase_correct_list = []
    for num in range(batch_num):
        batch_size = source_index[num].shape[0]
        for index in tqdm(range(batch_size)):
            retval_single, try_num, phase_correct = eval_decompose_accuracy_single(
                active_index, angle, attr, dif_generate, generate,
                mask, mixed_intensity, source_index, num, index
            )

            try_num_list.append(try_num)
            distance_rate.append(retval_single)
            phase_correct_list.append(phase_correct)

    distance_rate = np.concatenate(distance_rate)
    num = distance_rate.shape[0]

    output = {}
    output["Rp_under5"] = np.sum(distance_rate < 0.05) / num
    output["Rp_under10"] = np.sum(distance_rate < 0.10) / num
    output["Rp_under15"] = np.sum(distance_rate < 0.15) / num
    output["Rp_under20"] = np.sum(distance_rate < 0.2) / num
    output["Rp_under25"] = np.sum(distance_rate < 0.25) / num
    output["Rp_under30"] = np.sum(distance_rate < 0.30) / num
    output["phase_correct_rate"] = sum(phase_correct_list)/len(phase_correct_list)

    try_num_list = np.array(try_num_list)
    for _ in range(5):
        print("Predict " + str(_ + 1) + " phase: " + str(np.sum(try_num_list == _ + 1)/try_num_list.shape[0]))
        output["Predict " + str(_ + 1) + " phase"] = np.sum(try_num_list == _ + 1)/try_num_list.shape[0]
    print(output)
    return output

def eval_decompose_accuracy_single(
    active_index, angle, attr, dif_generate, generate, mask,
    mixed_intensity, source_index, batch_index, vis_index
):
    active_index = active_index[batch_index]
    angle = torch.rad2deg(angle[batch_index][0][vis_index])
    attr = attr[batch_index]
    dif_generate = dif_generate[batch_index]
    generate = generate[batch_index]
    mask = mask[batch_index]
    mixed_intensity = mixed_intensity[batch_index]
    source_index = source_index[batch_index][vis_index]
    correct_comp = torch.sum(source_index > -0.5)


    try_num = 0
    for i in range(len(active_index)):
        if active_index[i][vis_index]:
            try_num += 1

    attr = attr[vis_index]

    mask_this_index = []
    dif_generate_this_index = []
    generate_this_index = []
    mixed_intensity_this_index = []
    for i in range(try_num):
        mask_this_index.append(mask[i][vis_index])
        dif_generate_this_index.append(dif_generate[i][vis_index])
        generate_this_index.append(generate[i][vis_index])
        mixed_intensity_this_index.append(mixed_intensity[i][vis_index])

    generate_use_result = []
    generate_result_list = []
    dif_generate_result_list = []

    for i in range(try_num):
        generate_this_index[i][mask_this_index[i]] = 0
        generate_this_index[i][generate_this_index[i] < 0] = 0
        generate_result_list.append(np.array(generate_this_index[i]/(torch.max(generate_this_index[i])+1e-5)))

        dif_generate_this_index[i][mask_this_index[i]] = 0
        dif_generate_this_index[i][dif_generate_this_index[i] < 0] = 0
        dif_generate_result_list.append(np.array(dif_generate_this_index[i]/(torch.max(dif_generate_this_index[i]) + 1e-5)))
        if i == try_num - 1:
            generate_use_result.append(generate_result_list[-1])
        else:
            generate_use_result.append(dif_generate_result_list[-1])

    correct = []
    for i in range(correct_comp):
        attr[mask_this_index[0], :] = 0
        tmp = mixed_intensity_this_index[0] * attr[:, i]
        tmp[tmp < 0] = 0
        correct.append(np.array(tmp / (torch.max(tmp)+1e-5)))

    angle = np.array(angle)
    distance_matrix = np.ones((len(correct), len(generate_use_result))) * 10000
    distance_matrix_rate = np.zeros_like(distance_matrix)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if generate_use_result[j] is not None:
                distance_matrix[i, j], pattern, target_pattern = calc_distance(
                    angle, correct[i], angle, generate_use_result[j], "l2"
                )
                distance_matrix_rate[i, j] = distance_matrix[i, j]/np.linalg.norm(pattern)

    distance_mat = np.array(distance_matrix_rate)
    rp_list_tmp = []
    if try_num == correct_comp:
        distance_mat = np.array(distance_mat)
        for k in range(correct_comp):
            min_value = np.min(distance_mat)
            min_index = np.unravel_index(np.argmin(distance_mat), distance_mat.shape)
            rp_list_tmp.append(min_value)
            distance_mat[min_index[0], :] = np.inf
            distance_mat[:, min_index[1]] = np.inf
        phase_correct = 1
    else:
        distance_mat = np.array(distance_mat)
        rp_list_tmp = []
        for k in range(min(correct_comp, try_num)):
            min_value = np.min(distance_mat)
            min_index = np.unravel_index(np.argmin(distance_mat), distance_mat.shape)
            rp_list_tmp.append(min_value)
            distance_mat[min_index[0], :] = np.inf
            distance_mat[:, min_index[1]] = np.inf
        for k in range(max(correct_comp - try_num, 0)):
            rp_list_tmp.append(100)
        phase_correct = 0

    return np.array(rp_list_tmp), try_num, phase_correct