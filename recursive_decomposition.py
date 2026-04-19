from Model import utils
from Model import model
from calc_evaluation_metrics import eval_decompose_accuracy
import torch
from tqdm import tqdm
import pickle
from torch_geometric.data import Data, Batch
import numpy as np
import yaml
import copy
from Dataloader import dataloader
from DatabaseMatching import sandman
import math
import os
import time
import json

def generation(
        model, angle, intensity_input, mask,
        encoder_param_mixed, encoder_param_noisy,
        diffusion_param, device, search_result_index=None,
        database_name=None, confidence=None):
    model.eval()
    this_batch_size = angle.shape[0]
    t = torch.tensor(diffusion_param["t_max"]-1)

    ddpm = utils.DDPM(encoder_param_mixed, encoder_param_noisy, diffusion_param)

    intensity_noisy = torch.randn(this_batch_size, intensity_input.shape[-1]) * torch.sqrt(1-ddpm.alpha_bars[-1])

    pattern_generate = ddpm.denoising_process(
        model, angle, intensity_input, intensity_noisy, mask, t, device=device,
        search_result_index=search_result_index, database_name=database_name,
        confidence=confidence
    )

    return pattern_generate

def select_indices(batch, rate):
    batch_size = torch.max(batch) + 1
    select_indices = []
    offset = 0
    for i in range(batch_size):
        num_peaks = torch.sum(batch == i)
        if rate != 1.0:
            select_index = torch.randint(
                0, num_peaks, size=(int(num_peaks * rate),),
                device=batch.device)
        else:
            select_index = torch.arange(num_peaks, device=batch.device)
        select_index = torch.unique(select_index)
        select_indices.append(select_index + offset)
        offset += num_peaks
    select_indices = torch.concat(select_indices)
    return select_indices

def get_batch_detected(angle):
    batch_detected = []
    batch_idx = 0
    previous_value = 0
    for i in range(angle.shape[0]):
        if angle[i] - previous_value < 0:
            batch_idx += 1
        previous_value = angle[i]
        batch_detected.append(batch_idx)
    return torch.tensor(batch_detected, device=angle.device)


def load_model(path_config, path_model, path_database_test=None):
    with open(path_config, 'r') as f:
        cfg = yaml.safe_load(f)
    mixed_encoder_param = cfg["Encoder"]
    noisy_encoder_param = cfg["EncoderNoisy"]
    inject_layer_param = cfg["InjectLayer"]
    decoder_param = cfg["Decoder"]
    time_encoder_param = cfg["TimeEncoder"]
    diffusion_param = cfg["Diffusion"]
    optimizer_param = cfg["Optimizer"]
    loss_param = cfg["Loss"]
    path_database = cfg["path_database"]
    if path_database_test is not None:
        path_database["test"] = path_database_test

    best_model = model.ModelPTL.load_from_checkpoint(
        path_model,
        encoder_param_mixed=mixed_encoder_param,
        encoder_param_noisy=noisy_encoder_param,
        inject_layer_param=inject_layer_param,
        decoder_param=decoder_param,
        time_encoder_param=time_encoder_param,
        diffusion_param=diffusion_param,
        optimizer_params=optimizer_param,
        loss_param=loss_param,
        path_database=path_database,
        path_of_save_folder=None)

    return best_model, cfg


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generate_gaussian_convolution_mat(angle, fwhm=0.3, angle_min=0, angle_max=60, dim=2000):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    angle_all = np.linspace(angle_min, angle_max, dim)

    gaussian_convolution_mat = np.array(
        [gaussian(angle_all, np.array(angle[i]), sigma) for i in range(angle.shape[0])]).T

    return torch.tensor(gaussian_convolution_mat)


def normalize(intensity, threshold, mask, normalize=True):
    mask[intensity < threshold] = True
    intensity[mask] = 0
    if normalize:
        norm = torch.max(intensity, dim=-1, keepdim=True).values + 1e-5
        intensity = intensity / norm
        return intensity, mask, norm
    return intensity, mask


def recursive_generation_single_batch(
    batch_data, path_config, path_model,
    device, max_iter, database_name,
    perturbation_dict=None,
    candidates_num=15, threshold_rp=0.1,
    path_database=None, exclude_index=None,
):
    this_peak_num = batch_data.x_detected.shape[0]

    if perturbation_dict is not None:
        start = time.time()
        batch_detected = get_batch_detected(batch_data["x_detected"])
        if perturbation_dict["delete_peak"] != 0:
            delete_index = select_indices(batch_detected, perturbation_dict["delete_peak"])
            all_index = torch.arange(this_peak_num, device=batch_data.x_detected.device)
            use_index = all_index[torch.sum(all_index.unsqueeze(0) == delete_index.unsqueeze(1), dim=0) == 0]

            batch_data.x_detected = batch_data.x_detected[use_index]
            batch_data.intensity_detected_raw = batch_data.intensity_detected_raw[use_index]
            batch_data.attr_detected = batch_data.attr_detected[use_index]
            batch_data.batch = batch_data.batch[use_index]
            batch_detected = batch_detected[use_index]

            ptr = torch.tensor(
                [torch.sum(batch_detected < i) for i in range(batch_data.ptr.shape[0])]
            )
            batch_data.ptr = ptr

            batch_data._slice_dict["x_detected"] = ptr
            batch_data._slice_dict["intensity_detected_raw"] = ptr
            batch_data._slice_dict["attr_detected"] = ptr

        if perturbation_dict["perturbation_pos"] != 0:
            target_index = select_indices(batch_detected, perturbation_dict["perturbation_pos"])

            batch_data.x_detected[target_index] = batch_data.x_detected[target_index] + \
                                         torch.randn_like(batch_data.x_detected[target_index]) * \
                                                  perturbation_dict["perturbation_pos_scale"]

        if perturbation_dict["perturbation_intensity"] != 0:
            target_index = select_indices(batch_detected, perturbation_dict["perturbation_intensity"])

            batch_data.intensity_detected_raw[target_index] = \
                batch_data.intensity_detected_raw[target_index] + \
                torch.randn_like(batch_data.intensity_detected_raw[target_index]) * \
                batch_data.intensity_detected_raw[target_index] * \
                perturbation_dict["perturbation_intensity_scale"]

        print("perturbation time")
        print(time.time()-start)

    print("Loading Model")
    model, cfg = load_model(path_config, path_model, path_database_test=path_database)
    model.model = model.model.to(device)

    threshold = cfg["Diffusion"]["threshold"]

    if hasattr(batch_data, "x_detected"):
        angle, mask = utils.padding_from_batch(batch_data, "x_detected", generate_mask=True)
        angle = torch.deg2rad(angle)
        mixed_intensity, _ = utils.padding_from_batch(batch_data, "intensity_detected_raw")
        attr, _ = utils.padding_from_batch(batch_data, "attr_detected")
    else:
        raise NotImplementedError

    mixed_intensity[mask] == 0
    conv_mat = torch.stack(
        [generate_gaussian_convolution_mat(np.array(torch.rad2deg(angle[_]))) for _ in range(angle.shape[0])]
    )
    input_patterns = (conv_mat @ mixed_intensity.unsqueeze(-1))[:, :, 0]

    mixed_intensity_list = []
    angle_list = []
    mask_list = []
    generate_list = []
    dif_generate_list = []
    active_index_list = []
    search_result_list = []
    confidence_list = []

    active_index = torch.ones(angle.shape[0], dtype=torch.bool)
    mixed_intensity, mask, _ = normalize(mixed_intensity, threshold, mask)

    iter = 0
    while torch.sum(active_index) != 0 and iter < max_iter:
        mixed_intensity, mask, norm = normalize(
            mixed_intensity, threshold, mask)

        active_index_list.append(copy.deepcopy(active_index.detach().cpu()))
        mixed_intensity_list.append(copy.deepcopy(mixed_intensity.detach().cpu()))
        mask_list.append(copy.deepcopy(mask.detach().cpu()))
        angle_list.append(copy.deepcopy(angle.detach().cpu()))

        generate = copy.deepcopy(mixed_intensity)
        angle_use = angle[active_index]
        mixed_intensity_use = mixed_intensity[active_index]
        mask_use = mask[active_index]
        source_index = torch.stack(
            [data["source_index"][0, :] for data in batch_data.to_data_list()]
        )
        source_index_use = source_index[active_index]

        search_result_use = []
        ref_angle = copy.deepcopy(
            model.model.model_dif_transformer.database_angle[database_name + "_" + str(device)]
        ).to("cpu")
        ref_intensity = copy.deepcopy(
            model.model.model_dif_transformer.database_intensity[database_name + "_" + str(device)]
        ).to("cpu")
        ref_mask = copy.deepcopy(
            model.model.model_dif_transformer.database_mask[database_name + "_" + str(device)]
        ).to("cpu")

        ref_angle_list = []
        ref_intensity_list = []
        for ref_idx in range(ref_mask.shape[0]):
            ref_angle_list.append(torch.rad2deg(ref_angle[ref_idx][torch.logical_not(ref_mask[ref_idx])]))
            ref_intensity_list.append(ref_intensity[ref_idx][torch.logical_not(ref_mask[ref_idx])])

        print("Searching...")
        for obs_idx in tqdm(range(angle_use.shape[0])):
            angle_obs = torch.rad2deg(copy.deepcopy(angle_use[obs_idx][torch.logical_not(mask_use[obs_idx])]).to("cpu"))
            search_result_use.append(
                sandman.perform_search(
                    ref_angle_list, ref_intensity_list, angle_obs
                )
            )
        search_result_use = torch.tensor(search_result_use, device=angle_use.device)

        if exclude_index != "no":
            if exclude_index == "all":
                search_result = []

                for i in range(search_result_use.shape[0]):
                    tmp = search_result_use[i]
                    search_result.append(tmp[torch.sum(source_index_use[i].unsqueeze(-1) == tmp.unsqueeze(0), dim=0) == 0][:candidates_num])

                search_result = torch.stack(search_result)
            else:
                search_result = []

                for i in range(search_result_use.shape[0]):
                    tmp = search_result_use[i]
                    search_result.append(
                        tmp[torch.sum(source_index_use[i][exclude_index].unsqueeze(-1) == tmp.unsqueeze(0), dim=0) == 0][:candidates_num])

                search_result = torch.stack(search_result)
        else:
            search_result = search_result_use[:, :candidates_num]

        search_result_list.append(search_result)

        mixed_pattern_cls = model.model.model_dif_transformer.encoder_mixed(
            angle_use.to(torch.float32).to(device), mixed_intensity_use.to(torch.float32).to(device), mask_use.to(device), all_return=False
        )

        this_database_angle = model.model.model_dif_transformer.database_angle["test_" + str(mixed_pattern_cls.device)][
                                search_result.to(torch.int), :]
        this_database_mask = model.model.model_dif_transformer.database_mask["test_" + str(mixed_pattern_cls.device)][
                                search_result.to(torch.int), :]
        this_database_intensity = model.model.model_dif_transformer.database_intensity["test_" + str(mixed_pattern_cls.device)][
                                search_result.to(torch.int), :]
        this_database_angle = this_database_angle.reshape(-1, this_database_angle.shape[-1])
        this_database_intensity = this_database_intensity.reshape(-1, this_database_angle.shape[-1])
        this_database_mask = this_database_mask.reshape(-1, this_database_angle.shape[-1])

        database_pattern_cls = model.model.model_dif_transformer.encoder_database(
            this_database_angle.to(device), this_database_intensity.to(device), this_database_mask.to(device), all_return=False
        )
        database_pattern_embedding = database_pattern_cls.reshape(
            -1, candidates_num, database_pattern_cls.shape[-1]).contiguous()
        weights_database = torch.nn.functional.sigmoid(torch.bmm(
            database_pattern_embedding.contiguous(), mixed_pattern_cls.unsqueeze(-1).contiguous()
        )[:, :, 0] / math.sqrt(database_pattern_embedding.shape[-1]))

        confidence_list.append(weights_database.detach().cpu())

        use_weights, index = torch.max(weights_database, dim=-1)
        use_weights = use_weights.reshape(-1, 1).to(device)
        use_weights[use_weights < 0.5] = 0

        search_result = search_result.to(device)
        index = index.to(device)
        search_result_choose = search_result[
            torch.arange(index.shape[0], device=use_weights.device), index].reshape(-1, 1)

        print("Try " + str(iter+1) + ", Active: " + str(torch.sum(active_index)))
        generate_use = generation(
            model.model.to(device), angle_use.to(device),
            mixed_intensity_use.to(device), mask_use.to(device),
            cfg["Encoder"], cfg["EncoderNoisy"],
            cfg["Diffusion"], device,
            search_result_index=search_result_choose.to(device),
            database_name="test",
            confidence=use_weights
        )
        generate = generate.detach().cpu().to(torch.float32)
        generate[active_index] = generate_use.detach().cpu().to(torch.float32)

        dif_generate = mixed_intensity - generate

        generate = generate*norm
        dif_generate = dif_generate*norm

        generate[mask] = 0
        dif_generate[mask] = 0
        generate[generate < 0] = 0
        dif_generate[dif_generate < 0] = 0

        standard = [(conv_mat @ dif_generate_list[_].unsqueeze(-1))[:, :, 0] for _ in range(len(dif_generate_list))]
        standard.append((conv_mat@generate.unsqueeze(-1))[:, :, 0])
        standard = torch.stack(standard, dim=1)
        fitted_result = (standard.transpose(-1, -2) @ torch.linalg.pinv(standard.transpose(-1, -2)) @ input_patterns.unsqueeze(-1))[:, :, 0]
        rp = torch.linalg.norm(fitted_result-input_patterns, dim=-1)/torch.linalg.norm(input_patterns, dim=-1)

        generate_list.append(copy.deepcopy(generate.detach().cpu()))
        dif_generate_list.append(copy.deepcopy(dif_generate.detach().cpu()))

        mixed_intensity = copy.deepcopy(generate)

        active_index[dif_generate.max(dim=-1).values < threshold] = False
        active_index[rp < threshold_rp] = False

        iter += 1

    return mixed_intensity_list, angle_list, mask_list, generate_list, dif_generate_list, active_index_list, attr, search_result_list, confidence_list


def recursive_generation(
    path_data, path_config, path_model,
    device, max_iter, batch_size, path_save_dir,
    max_batch_num = 1000, perturbation_dict=None,
    candidates_num=None,
    path_database=None, exclude_index="no"
):
    os.makedirs(path_save_dir, exist_ok=True)
    print(path_config)
    with open(path_config, 'r') as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    cfg["Dataloader"]["path_test"] = path_data
    cfg["Dataloader"]["batch_size"] = batch_size

    data_config = cfg["Dataloader"]
    datamodule = dataloader.DataModule(**data_config)

    mixed_intensity_list = []
    angle_list = []
    mask_list = []
    generate_list = []
    dif_generate_list = []
    active_index_list = []
    attr_list = []
    source_index_list = []
    search_result_list = []
    confidence_list = []

    batch_num = 0
    for batch in datamodule.test_dataloader():
        if batch_num < max_batch_num:
            source_index = torch.stack(
                [data["source_index"][0, :] for data in batch.to_data_list()]
            )
            source_index_list.append(source_index.to("cpu"))

            mixed_intensity, angle, mask, \
                generate, dif_generate, active_index, attr, search_result, confidence \
                = recursive_generation_single_batch(
                    batch, path_config, path_model,
                    device, max_iter,
                    database_name="test", perturbation_dict=perturbation_dict,
                    candidates_num=candidates_num,
                    path_database=path_database, exclude_index=exclude_index)

            confidence_list.append(confidence)
            mixed_intensity_list.append(mixed_intensity)
            angle_list.append(angle)
            mask_list.append(mask)
            generate_list.append(generate)
            dif_generate_list.append(dif_generate)
            active_index_list.append(active_index)
            attr_list.append(attr)
            search_result_list.append(search_result)

        batch_num += 1

    with open(path_save_dir + "/mixed_intensity.pkl", "wb") as f:
        pickle.dump(mixed_intensity_list, f)
    with open(path_save_dir + "/angle.pkl", "wb") as f:
        pickle.dump(angle_list, f)
    with open(path_save_dir + "/mask.pkl", "wb") as f:
        pickle.dump(mask_list, f)
    with open(path_save_dir + "/generate.pkl", "wb") as f:
        pickle.dump(generate_list, f)
    with open(path_save_dir + "/dif_generate.pkl", "wb") as f:
        pickle.dump(dif_generate_list, f)
    with open(path_save_dir + "/active_index.pkl", "wb") as f:
        pickle.dump(active_index_list, f)
    with open(path_save_dir + "/attr.pkl", "wb") as f:
        pickle.dump(attr_list, f)
    with open(path_save_dir + "/source_index.pkl", "wb") as f:
        pickle.dump(source_index_list, f)
    with open(path_save_dir + "/search_result.pkl", "wb") as f:
        pickle.dump(search_result_list, f)
    with open(path_save_dir + "/confidence.pkl", "wb") as f:
        pickle.dump(confidence_list, f)


    output = eval_decompose_accuracy(path_save_dir)

    with open(path_save_dir + "/metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    path_data = "Data/MP/cubic_2phase/test.pkl"

    path_model = "result/phasedifformer/confidence_estimator/version_0/model_checkpoint/best.ckpt"
    path_config = "result/phasedifformer/confidence_estimator/config_use.yaml"

    device = "cuda:0"
    max_iter = 5
    batch_size = 100
    path_save_dir = "inference/cubic_2phase"

    recursive_generation(
        path_data, path_config, path_model,
        device, max_iter, batch_size, path_save_dir,
        max_batch_num = 3,
        candidates_num=15, exclude_index="all"
    )
