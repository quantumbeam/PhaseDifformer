import numpy as np
import torch
import copy
import os
import glob
import pickle
from tqdm import tqdm

def select_indices(batch, rate):
    if rate == 1.0:
        return torch.arange(batch.shape[0], device=batch.device)
    mask = torch.rand(batch.shape[0], device=batch.device) < rate
    return torch.nonzero(mask, as_tuple=False).flatten()

def get_batch_detected(angle):
    resets = (angle[1:] < angle[:-1]).to(torch.int64)
    resets = torch.cat([torch.zeros(1, dtype=torch.int64, device=angle.device), resets])
    return torch.cumsum(resets, dim=0)

def perturbation(angle, intensity, perturbation_dict):
    this_peak_num = intensity.shape[0]
    if perturbation_dict is not None:
        if perturbation_dict["delete_peak"] != 0:
            delete_index = select_indices(torch.zeros_like(angle), perturbation_dict["delete_peak"])
            all_index = torch.arange(this_peak_num, device=angle.device)
            use_index = all_index[torch.sum(all_index.unsqueeze(0) == delete_index.unsqueeze(1), dim=0) == 0]

            angle = angle[use_index]
            intensity = intensity[use_index]

        if perturbation_dict["perturbation_pos"] != 0:
            target_index = select_indices(torch.zeros_like(angle), perturbation_dict["perturbation_pos"])
            angle[target_index] = angle[target_index] + \
                                        torch.randn_like(angle[target_index]) * \
                                        perturbation_dict["perturbation_pos_scale"]

        if perturbation_dict["perturbation_intensity"] != 0:
            target_index = select_indices(torch.zeros_like(angle), perturbation_dict["perturbation_intensity"])

            intensity[target_index] = \
                intensity[target_index] + \
                torch.randn_like(intensity[target_index]) * intensity[target_index] * perturbation_dict["perturbation_intensity_scale"]
    return angle, intensity

def angle_to_d(theta_deg, wavelength=1.5406):
    theta_array = np.array(theta_deg, dtype=float)

    theta_rad = (theta_array) * (np.pi / 180.0)

    d_values = wavelength / (2.0 * np.sin(theta_rad))

    return d_values

def lorentz_prob(mismatch, sigma):

    return 1.0 / (1.0 + (mismatch / sigma)**2)

def weighted_peak_position_score(
    angle_ref,
    I_ref,
    angle_obs,
    sigma_d=1.0,
    min_prob=0.0001,
    convert_d = False
):
    angle_ref = np.array(angle_ref, dtype=float)
    I_ref = np.array(I_ref, dtype=float)
    angle_obs = np.array(angle_obs, dtype=float)

    if convert_d:
        d_ref = angle_to_d(angle_ref)
        d_obs = angle_to_d(angle_obs)
    else:
        d_ref = angle_ref
        d_obs = angle_obs

    if len(angle_obs) == 0:
        return 1e-10

    if np.sum(I_ref) > 0:
        I_ref_norm = I_ref / np.sum(I_ref)
    else:
        I_ref_norm = np.ones_like(I_ref) / len(I_ref)

    log_score_sum = 0.0
    for d_r, w_r in zip(d_ref, I_ref_norm):
        idx_min = np.argmin(np.abs(d_obs - d_r))
        d_o = d_obs[idx_min]

        mismatch = abs(d_r - d_o)
        p_hit = lorentz_prob(mismatch, sigma_d)
        p_hit = max(p_hit, min_prob)

        log_score_sum += w_r * np.log(p_hit)

    total_score = np.exp(log_score_sum)
    return total_score

def load_pkl(path_pkl):
    with open(path_pkl, mode='br') as fi:
        data = pickle.load(fi)
    return data

def perform_search(
    angle_ref_list,
    I_ref_list,
    angle_obs,
):
    score_list = []
    for i in range(len(angle_ref_list)):
        score_list.append(
            weighted_peak_position_score(
                angle_ref_list[i], I_ref_list[i], angle_obs
            )
        )
    score_list = np.array(score_list)
    return np.argsort(-score_list)[:20]


def add_search_result(path_data, path_database, path_save, database_size, threshold, perturbation_dict=None, max_num=10000000):
    print("Loading data...")
    data_all = load_pkl(path_data)
    data_num = len(data_all)

    database_all = load_pkl(path_database)

    print("Data Num: " + str(data_num))
    print("Database Num: " + str(len(database_all)))

    print("Preparing Database...")
    for i in range(len(database_all)):
        angle_detected_tmp = database_all[i]["angle_detected"]
        intensity_detected_tmp = database_all[i]["peak_detected"]

        angle_detected_tmp = angle_detected_tmp[intensity_detected_tmp > threshold]
        intensity_detected_tmp = intensity_detected_tmp[intensity_detected_tmp > threshold]

        database_all[i]["angle_detected"] = angle_detected_tmp
        database_all[i]["peak_detected"] = intensity_detected_tmp
    print("Complete")

    source_index_list = []
    search_result_index_list = []


    for i in tqdm(range(data_num)):
        if i < max_num:
            data = data_all[i]

            search_result = torch.arange(len(database_all))
            source_index = data["source_index"][0]
            indices = torch.randperm(search_result.size(0))[:database_size]
            search_result = search_result[indices]
            for k in range(source_index.shape[0]):
                if source_index[k] > -0.5:
                    if torch.sum(search_result == source_index[k]) == 0:
                        search_result[k] = source_index[k]

            angle = data["x_detected"]
            intensity_detected = data["intensity_detected_raw"]

            angle, intensity_detected = perturbation(angle, intensity_detected, perturbation_dict)

            angle = angle[intensity_detected > threshold]
            intensity_detected = intensity_detected[intensity_detected > threshold]

            search_result = np.array(search_result)
            score_list = []
            for idx_ref in range(search_result.shape[0]):
                score = weighted_peak_position_score(
                    database_all[search_result[idx_ref]]["angle_detected"],
                    database_all[search_result[idx_ref]]["peak_detected"],
                    angle
                )
                score_list.append(score)
            score_list = np.array(score_list)

            source_index_list.append(np.array(source_index))
            search_result_index_list.append(search_result[np.argsort(-score_list)])

    source_index_list = torch.tensor(np.array(source_index_list))
    search_result_index_list = torch.tensor(np.array(search_result_index_list))

    for i in range(20):
        search_result_index_list_tmp = search_result_index_list[:, :i+1]
        hit = torch.sum(source_index_list.unsqueeze(-1) == search_result_index_list_tmp.unsqueeze(-2), dim=(-1, -2))

        print("Top " + str(i+1) + " accuracy: " + str(torch.sum(hit > 0)/hit.shape[0]))

    new_data = []
    for i in tqdm(range(len(data_all))):
        data = data_all[i]
        data.search_result = torch.tensor(search_result_index_list[i])[:10].unsqueeze(0).reshape(1, -1)
        new_data.append(data)

    print(new_data[0])

    if path_save is not None:
        with open(path_save, mode='wb') as fo:
            pickle.dump(new_data, fo)