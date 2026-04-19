import torch
from . import encoder
from . import inject_layer
from . import decoder
from . import CrossAttentionModule
from torch.optim import lr_scheduler
import pytorch_lightning as ptl
import os
import shutil
import numpy as np
from Loss import losses
from Model import utils
import math
import copy
from tqdm import tqdm


class ModelDifTransFormer(torch.nn.Module):
    def __init__(
        self,
        encoder_param_mixed,
        encoder_param_noisy,
        inject_layer_param,
        decoder_param,
        time_encoder_param,
        diffusion_param,
        path_database
    ):
        super(ModelDifTransFormer, self).__init__()
        if encoder_param_mixed["encoder_name"] != "confidence_estimator":
            self.encoder_noisy = encoder.Encoder(
                **encoder_param_noisy
            )
        else:
            self.encoder_noisy = None

        self.inject_mixed = inject_layer_param["inject_mixed"]

        self.encoder_name = encoder_param_mixed["encoder_name"]
        self.encoder_param_mixed = encoder_param_mixed[self.encoder_name]
        print("Injecting Mixed pattern: " + str(self.encoder_name) + " Encoder")

        if self.encoder_name == "without_guidance":
            self.encoder_single = encoder.Encoder(**encoder_param_mixed[self.encoder_name]["single"])

        elif self.encoder_name == "with_guidance":
            self.encoder_single = encoder.Encoder(**encoder_param_mixed[self.encoder_name]["single"])
            self.encoder_mixed_pretransform = encoder.Encoder(
                **encoder_param_mixed[self.encoder_name]["single"])
            self.encoder_mixed = CrossAttentionModule.TransformerEncoderLayer_SelfAndCrossAttention(
                layer_order_for_src1=encoder_param_mixed[self.encoder_name]["mixed"]["layer_order_for_src1"],
                layer_order_for_src2=encoder_param_mixed[self.encoder_name]["mixed"]["layer_order_for_src2"],
                d_model=encoder_param_mixed[self.encoder_name]["mixed"]["d_model"],
                nhead=encoder_param_mixed[self.encoder_name]["mixed"]["nhead"],
                dim_feedforward=encoder_param_mixed[self.encoder_name]["mixed"]["dim_feedforward"],
                dropout=encoder_param_mixed[self.encoder_name]["mixed"]["dropout"],
                layer_norm_eps=encoder_param_mixed[self.encoder_name]["mixed"]["layer_norm_eps"],
                norm_first=encoder_param_mixed[self.encoder_name]["mixed"]["norm_first"],
            )

        elif self.encoder_name == "confidence_estimator":
            self.num_use = encoder_param_mixed[self.encoder_name]["num_use"]
            if self.encoder_param_mixed["weight_method"] == "dot_product_sigmoid_hard_switch":
                self.encoder_database = encoder.Encoder_w_cls(**encoder_param_mixed[self.encoder_name]["single"])
                self.encoder_mixed = encoder.Encoder_w_cls(**encoder_param_mixed[self.encoder_name]["single"])
            else:
                raise NotImplementedError()

            self.guided_noise_predictor, _ = utils.load_model(
                self.encoder_param_mixed["with_guidance_config"],
                self.encoder_param_mixed["with_guidance_model"],
                path_database=path_database
            )

            self.without_guided_noise_predictor, _ = utils.load_model(
                self.encoder_param_mixed["without_guidance_config"],
                self.encoder_param_mixed["without_guidance_model"],
                path_database=path_database
            )
        else:
            raise NotImplementedError()

        if self.encoder_name != "confidence_estimator":
            self.inject_layer = inject_layer.InjectLayer(
                **inject_layer_param
            )

            self.decoder = decoder.Decoder(
                **decoder_param
            )
            self.time_encoder = utils.MLP(**time_encoder_param)
        else:
            self.inject_layer = None
            self.decoder = None
            self.time_encoder = None

        self.database = {}
        self.database_angle = {}
        self.database_intensity = {}
        self.database_mask = {}

        if self.encoder_name != "without_guidance":
            device_count = torch.cuda.device_count()
            device_ids = [f'cuda:{id}' for id in range(device_count)]
            for key in path_database:
                for device in device_ids:
                    self.database[key] = utils.load_pkl(path_database[key])
                    database_angle_list = []
                    database_intensity_list = []
                    for i in tqdm(range(len(self.database[key]))):
                        database_angle_list.append(torch.deg2rad(torch.tensor(self.database[key][i]["angle_detected"])))
                        database_intensity_list.append(torch.tensor(self.database[key][i]["peak_detected"]))
                    database_angle_list, mask = utils.padding_from_list(database_angle_list, generate_mask=True)
                    database_intensity_list, _ = utils.padding_from_list(database_intensity_list, generate_mask=False)
                    mask[database_intensity_list < diffusion_param["threshold"]] = True
                    self.database_angle[key + "_" + device] = database_angle_list.to(device)
                    self.database_intensity[key + "_" + device] = database_intensity_list.to(device)
                    self.database_mask[key + "_" + device] = mask.to(device)
                    self.database_angle[key] = database_angle_list.to("cpu")
                    self.database_intensity[key] = database_intensity_list.to("cpu")
                    self.database_mask[key] = mask.to("cpu")

    def forward(
        self, angle, intensity_input, intensity_noisy, time, mask,
        search_result_index = None, database_name = None,
        confidence=None
    ):
        if self.time_encoder is not None:
            time_embedding = self.time_encoder(time.unsqueeze(-1).to(torch.float32))
        else:
            time_embedding = None
        this_batch_size = angle.shape[0]
        if self.inject_mixed:
            if self.encoder_name == "without_guidance":
                mixed_pattern_embedding = self.encoder_single(
                    angle.to(torch.float32), intensity_input.to(torch.float32), mask
                )
                weights = None

            elif self.encoder_name == "with_guidance":
                device = angle.device
                mixed_pattern_embedding = self.encoder_mixed_pretransform(
                    angle.to(torch.float32), intensity_input.to(torch.float32), mask
                ) #[B, Peak, d_model]

                this_database_angle = self.database_angle[database_name+"_"+str(device)][search_result_index.to(torch.int), :]
                this_database_mask = self.database_mask[database_name+"_"+str(device)][search_result_index.to(torch.int), :]
                this_database_intensity = self.database_intensity[database_name+"_"+str(device)][search_result_index.to(torch.int), :]
                this_database_angle = this_database_angle.reshape(-1, this_database_angle.shape[-1])
                this_database_intensity = this_database_intensity.reshape(-1, this_database_angle.shape[-1])
                this_database_mask = this_database_mask.reshape(-1, this_database_angle.shape[-1])

                single_pattern_embedding = self.encoder_single(
                    this_database_angle.to(torch.float32),
                    this_database_intensity.to(torch.float32),
                    this_database_mask
                ) #[B*use_num, Peak, d_model]

                #mixed pattern embedding: [B, Peak, d_model]
                #mask: [B, Peak]
                #single_pattern_embedding: [B*use_num, Peak, d_model]
                #this_database_mask: [B*use_num, Peak]

                mixed_pattern_embedding, _ = self.encoder_mixed(
                    mixed_pattern_embedding, single_pattern_embedding,
                    mask, this_database_mask
                ) #[B, peak_num, d_model]
                weights = None

            elif self.encoder_name == "confidence_estimator":
                device = angle.device
                search_num = search_result_index.shape[-1]
                with torch.no_grad():
                    token_num = angle.shape[-1]
                    time_for_guidance = time.unsqueeze(-1).repeat(1, search_num).flatten().contiguous()
                    angle_for_guidance = angle.unsqueeze(1).repeat(1, search_num, 1).reshape(-1, token_num).contiguous()
                    intensity_noisy_for_guidance = intensity_noisy.unsqueeze(1).repeat(1, search_num, 1).reshape(-1, token_num).contiguous()
                    intensity_input_for_guidance = intensity_input.unsqueeze(1).repeat(1, search_num, 1).reshape(-1, token_num).contiguous()
                    mask_for_guidance = mask.unsqueeze(1).repeat(1, search_num, 1).reshape(-1, token_num).contiguous()
                    search_result_index_for_guidance = search_result_index.flatten().reshape(-1, 1).contiguous()
                    guided_noise_prediction = self.guided_noise_predictor.model_dif_transformer(
                        angle_for_guidance, intensity_input_for_guidance, intensity_noisy_for_guidance,
                        time_for_guidance, mask_for_guidance, search_result_index_for_guidance, database_name
                    )[0]
                    without_guided_noise_prediction = self.without_guided_noise_predictor.model_dif_transformer(
                        angle, intensity_input, intensity_noisy, time, mask
                    )[0]
                    weights = confidence
                decode_result = guided_noise_prediction.to(torch.float32) * confidence.to(torch.float32) \
                                + without_guided_noise_prediction.to(torch.float32) * (1-confidence.to(torch.float32))
                add_embedding = None

            else:
                raise NotImplementedError()

            if time_embedding is not None:
                mixed_pattern_embedding = torch.concatenate(
                    [time_embedding.unsqueeze(1), mixed_pattern_embedding], dim=1)
                mask_for_injection = torch.concat([torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device), mask],dim=1)
                add_embedding = None

        else:
            add_embedding = time_embedding.unsqueeze(1)
            weights = None

        if self.encoder_noisy is not None:
            noisy_pattern_embedding = self.encoder_noisy(angle.to(torch.float32).contiguous(), intensity_noisy.to(torch.float32).contiguous(), mask.contiguous(), add_embedding=add_embedding)
        if add_embedding is None:
            if self.inject_layer is not None:
                embedding = self.inject_layer(
                    mixed_pattern_embedding.to(torch.float32).contiguous(), noisy_pattern_embedding.to(torch.float32).contiguous(),
                    mask_for_injection.contiguous(), mask.contiguous())
        elif self.encoder_noisy is not None:
            embedding = noisy_pattern_embedding

        if self.decoder is not None:
            decode_result = self.decoder(embedding.to(torch.float32))

        return decode_result, weights, search_result_index


class Model(torch.nn.Module):
    def __init__(
            self,
            encoder_param_mixed,
            encoder_param_noisy,
            inject_layer_param,
            decoder_param,
            time_encoder_param,
            diffusion_param,
            path_database
    ):
        super(Model, self).__init__()

        self.model_dif_transformer = ModelDifTransFormer(
            encoder_param_mixed,
            encoder_param_noisy,
            inject_layer_param,
            decoder_param,
            time_encoder_param,
            diffusion_param,
            path_database
        )

        self.ddpm = utils.DDPM(
            encoder_param_mixed,
            encoder_param_noisy,
            diffusion_param
        )

        self.t_max = diffusion_param["t_max"]
        self.components_min = diffusion_param["components_min"]
        self.threshold = diffusion_param["threshold"]
        self.noise_contained = diffusion_param["noise_contained"]

        self.encoder_param_mixed = encoder_param_mixed
        self.num_use = encoder_param_mixed[encoder_param_mixed["encoder_name"]].get("num_use")
        self.no_flip = diffusion_param.get("no_flip", False)
        self.choice_rate = encoder_param_mixed[encoder_param_mixed["encoder_name"]].get("choice_rate")

        if self.encoder_param_mixed["encoder_name"] == "confidence_estimator":
            self.accept_rate = self.encoder_param_mixed["confidence_estimator"].get("accept_rate", 1.0)

        print("No Flip:" + str(self.no_flip))


    def forward(self, batch_data, name=None):
        this_batch_size = batch_data.components_num.shape[0]
        if hasattr(batch_data, "x_detected"):
            padded_batch = utils.padding_from_batch_multi(
                batch_data, ["x_detected", "intensity_detected", "intensity_detected_raw", "attr_detected"], generate_mask=True)
            angle, mask = padded_batch["x_detected"]
            intensity, _ = padded_batch["intensity_detected"]
            intensity_detected, _ = padded_batch["intensity_detected_raw"]
            attr, _ = padded_batch["attr_detected"]
            search_result = batch_data["search_result"].reshape(this_batch_size, -1)
            source_index = batch_data["source_index"].reshape(this_batch_size, -1)
        else:
            raise NotImplementedError()

        angle = torch.deg2rad(angle)

        if self.encoder_param_mixed["encoder_name"] == "with_guidance":
            if self.noise_contained:
                noise_attr = attr[:, :, -1]
                attr = attr[:, :, :attr.shape[-1] - 1]
            else:
                noise_attr = None

            components_num = copy.deepcopy(batch_data.components_num)

            mask[intensity_detected < self.threshold] = True
            delete_index = torch.rand_like(source_index)
            delete_index[source_index < 0] = 10
            delete_index = torch.argmin(delete_index, dim=-1)
            search_result = source_index[torch.arange(source_index.shape[0], device=source_index.device), delete_index]
            search_result = search_result.unsqueeze(-1)

            attr[torch.arange(delete_index.shape[0], device=delete_index.device), :, delete_index] = 0
            attr_sum = torch.sum(attr, dim=-1)

            if noise_attr is not None:
                if not self.no_flip:
                    attr_sum[components_num == 1] = 1 - attr_sum[components_num == 1] - noise_attr[components_num == 1]
                intensity_input = intensity * (attr_sum + noise_attr)
            else:
                if not self.no_flip:
                    attr_sum[components_num == 1] = 1 - attr_sum[components_num == 1]
                intensity_input = intensity * attr_sum

            intensity = copy.deepcopy(intensity_detected)

        elif self.encoder_param_mixed["encoder_name"] == "without_guidance":
            if self.noise_contained:
                noise_attr = attr[:, :, -1]
                attr = attr[:, :, :attr.shape[-1] - 1]
            else:
                noise_attr = None

            components_num = copy.deepcopy(batch_data.components_num)

            mask[intensity_detected < self.threshold] = True
            delete_index = torch.rand_like(source_index)
            delete_index[source_index < 0] = 10
            delete_index = torch.argmin(delete_index, dim=-1)

            attr[torch.arange(delete_index.shape[0], device=delete_index.device), :, delete_index] = 0
            attr_sum = torch.sum(attr, dim=-1)

            if noise_attr is not None:
                if not self.no_flip:
                    attr_sum[components_num == 1] = 1 - attr_sum[components_num == 1] - noise_attr[components_num == 1]
                intensity_input = intensity * (attr_sum + noise_attr)
            else:
                if not self.no_flip:
                    attr_sum[components_num == 1] = 1 - attr_sum[components_num == 1]
                intensity_input = intensity * attr_sum
            intensity = copy.deepcopy(intensity_detected)

        elif self.encoder_param_mixed["encoder_name"] == "confidence_estimator":
            if self.noise_contained:
                noise_attr = attr[:, :, -1]
                attr = attr[:, :, :attr.shape[-1] - 1]
            else:
                noise_attr = None

            components_num = copy.deepcopy(batch_data.components_num)

            mask[intensity_detected < self.threshold] = True

            prob_mask = torch.zeros_like(search_result, dtype=torch.bool, device=search_result.device)
            prob_mask[torch.rand(size=(search_result.shape[0], search_result.shape[1]), device=prob_mask.device) > self.accept_rate] = True

            search_result_use = [
                search_result[i, torch.logical_not(torch.logical_and(prob_mask[i], torch.sum(source_index[i].unsqueeze(-1) == search_result[i].unsqueeze(0), dim=0) > 0))][:self.num_use]
                for i in range(search_result.shape[0])
            ]
            search_result = torch.stack(search_result_use)
            search_result = search_result[:, :self.num_use]

            mixed_pattern_cls = self.model_dif_transformer.encoder_mixed(
                angle.to(torch.float32), intensity_detected.to(torch.float32), mask, all_return=False
            )

            this_database_angle = self.model_dif_transformer.database_angle[name + "_" + str(mixed_pattern_cls.device)][
                                  search_result.to(torch.int), :]
            this_database_mask = self.model_dif_transformer.database_mask[name + "_" + str(mixed_pattern_cls.device)][
                                 search_result.to(torch.int), :]
            this_database_intensity = self.model_dif_transformer.database_intensity[name + "_" + str(mixed_pattern_cls.device)][
                                      search_result.to(torch.int), :]
            this_database_angle = this_database_angle.reshape(-1, this_database_angle.shape[-1])
            this_database_intensity = this_database_intensity.reshape(-1, this_database_angle.shape[-1])
            this_database_mask = this_database_mask.reshape(-1, this_database_angle.shape[-1])

            database_pattern_cls = self.model_dif_transformer.encoder_database(
                this_database_angle, this_database_intensity, this_database_mask, all_return=False
            )
            database_pattern_embedding = database_pattern_cls.reshape(
                -1, self.num_use, database_pattern_cls.shape[-1]).contiguous()
            weights_database = torch.nn.functional.sigmoid(torch.bmm(
                database_pattern_embedding.contiguous(), mixed_pattern_cls.unsqueeze(-1).contiguous()
            )[:, :, 0] / math.sqrt(database_pattern_embedding.shape[-1]))
            use_weights, index = torch.max(weights_database, dim=-1)
            use_weights = use_weights.reshape(-1, 1)
            search_result_choose = search_result[
                torch.arange(index.shape[0], device=use_weights.device), index].reshape(-1, 1)

            max_val, search_hit = torch.max(
                (source_index.unsqueeze(-1) == search_result_choose.unsqueeze(-2)).to(torch.int), dim=-1)
            search_hit[max_val == 0] = 1000000
            search_hit[source_index < 0] = 10000000

            delete_num = torch.argmin(
                search_hit.to(torch.float32) + torch.rand_like(search_hit.to(torch.float32)), dim=-1)

            attr[torch.arange(delete_num.shape[0], device=delete_num.device), :, delete_num] = 0
            attr_sum = torch.sum(attr, dim=-1)

            if noise_attr is not None:
                if not self.no_flip:
                    attr_sum[components_num == 1] = 1 - attr_sum[components_num == 1] - noise_attr[components_num == 1]
                intensity_input = intensity * (attr_sum + noise_attr)
            else:
                if not self.no_flip:
                    attr_sum[components_num == 1] = 1 - attr_sum[components_num == 1]
                intensity_input = intensity * attr_sum

            intensity = copy.deepcopy(intensity_detected)

        else:
            raise NotImplementedError()


        this_batch_size = intensity_input.shape[0]
        intensity_noisy, t_list, noise = self.ddpm.diffusion_process(intensity_input)

        if str(intensity_input.device) != "cpu":
            device_count = torch.cuda.device_count()
            n = min(device_count, this_batch_size)
            device_ids = [f'cuda:{id}' for id in range(n)]

            one_device_size = (this_batch_size // n)

            inputs = []

            if self.encoder_param_mixed["encoder_name"] == "without_guidance":
                for i in range(n):
                    inputs.append((
                        angle[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        intensity[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        intensity_noisy[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        t_list[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        mask[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                    ))

            elif self.encoder_param_mixed["encoder_name"] == "with_guidance":
                for i in range(n):
                    inputs.append((
                        angle[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        intensity[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        intensity_noisy[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        t_list[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        mask[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        search_result[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        name
                    ))

            elif self.encoder_param_mixed["encoder_name"] == "confidence_estimator":
                for i in range(n):
                    inputs.append((
                        angle[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        intensity[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        intensity_noisy[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        t_list[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        mask[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        search_result_choose[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i]),
                        name,
                        use_weights[one_device_size * i:one_device_size * (i + 1)].to(device_ids[i])
                    ))

            else:
                raise NotImplementedError()

            device_ids = [f'cuda:{id}' for id in range(n)]
            replicas = torch.nn.parallel.replicate(self.model_dif_transformer, device_ids)

            x = torch.nn.parallel.parallel_apply(replicas, inputs)

            output = torch.nn.parallel.gather(x, intensity_input.device)
            output, weights, search_result_index = output

        else:
            raise NotImplementedError()

        if self.encoder_param_mixed["encoder_name"] == "confidence_estimator":
            return output, noise, mask, weights_database, source_index, search_result
        else:
            return output, noise, mask, weights, source_index, search_result_index


class ModelPTL(ptl.LightningModule):
    def __init__(
            self,
            encoder_param_mixed,
            encoder_param_noisy,
            inject_layer_param,
            decoder_param,
            time_encoder_param,
            diffusion_param,
            optimizer_params,
            loss_param,
            path_database,
            path_of_save_folder
    ):
        super(ModelPTL, self).__init__()
        self.model = Model(
            encoder_param_mixed,
            encoder_param_noisy,
            inject_layer_param,
            decoder_param,
            time_encoder_param,
            diffusion_param,
            path_database
        )
        self.optimizer_params = optimizer_params
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.path_of_save_folder = path_of_save_folder
        self.epoch_now = 0
        self.loss_param = loss_param

        self.encoder_mixed_name = encoder_param_mixed["encoder_name"]

        self.automatic_optimization = False
        self.step=1

        self.lambda_noise = loss_param.get('lambda_noise')
        self.lambda_inference = loss_param.get('lambda_inference')

        self.perturbation_pos = loss_param.get('perturbation_pos', 0)
        self.perturbation_pos_scale = loss_param.get('perturbation_pos_scale')
        self.perturbation_intensity = loss_param.get('perturbation_intensity', 0)
        self.perturbation_intensity_scale = loss_param.get('perturbation_intensity_scale')
        self.delete_peak = loss_param.get('delete_peak', 0)

        print("perturbation_pos: " + str(self.perturbation_pos))
        print("perturbation_pos_scale: " + str(self.perturbation_pos_scale))
        print("perturbation_intensity: " + str(self.perturbation_intensity))
        print("perturbation_intensity_scale: " + str(self.perturbation_intensity_scale))
        print("delete_peak: " + str(self.delete_peak))

        if self.perturbation_pos == 0 and self.perturbation_intensity == 0 and self.delete_peak == 0:
            self.perturbation_dict = None
        else:
            self.perturbation_dict = {
                "delete_peak": self.delete_peak,
                "perturbation_pos": self.perturbation_pos,
                "perturbation_pos_scale": self.perturbation_pos_scale,
                "perturbation_intensity": self.perturbation_intensity,
                "perturbation_intensity_scale": self.perturbation_intensity_scale
            }
        print(self.perturbation_dict)


    def forward(self, batch_data, name):
        this_peak_num = batch_data.x_detected.shape[0]
        if self.perturbation_dict is not None:
            batch_detected = utils.get_batch_detected(batch_data["x_detected"])
            if self.perturbation_dict["delete_peak"] != 0:
                delete_index = utils.select_indices(batch_detected, self.perturbation_dict["delete_peak"])
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

            if self.perturbation_dict["perturbation_pos"] != 0:
                target_index = utils.select_indices(batch_detected, self.perturbation_dict["perturbation_pos"])

                batch_data.x_detected[target_index] = batch_data.x_detected[target_index] + \
                                                      torch.randn_like(batch_data.x_detected[target_index]) * \
                                                      self.perturbation_dict["perturbation_pos_scale"]

            if self.perturbation_dict["perturbation_intensity"] != 0:
                target_index = utils.select_indices(batch_detected, self.perturbation_dict["perturbation_intensity"])

                batch_data.intensity_detected_raw[target_index] = \
                    batch_data.intensity_detected_raw[target_index] + \
                    torch.randn_like(batch_data.intensity_detected_raw[target_index]) * \
                    batch_data.intensity_detected_raw[target_index] * \
                    self.perturbation_dict["perturbation_intensity_scale"]

        model_output, noise, mask, weights, source_index, search_result = self.model(batch_data, name)
        this_batch_size = model_output.shape[0]

        if self.loss_param["mode"] == "mse":
            loss = losses.mse_loss(model_output.to(torch.float32), noise.to(torch.float32), mask)
            if search_result is not None:
                weights_truth = torch.sum(source_index.unsqueeze(1) == search_result.unsqueeze(-1), dim=-1)
                acc = torch.sum(torch.sum(weights_truth, dim=-1) > 0) / weights_truth.shape[0]
            else:
                acc=0
            retval = {
                "loss": loss,
                "acc": acc,
                "seen": this_batch_size
            }
        elif self.loss_param["mode"] == "mse_crossentropy":
            diffusion_loss = losses.mse_loss(model_output.to(torch.float32), noise.to(torch.float32), mask)
            weights_truth = torch.sum(source_index.unsqueeze(1) == search_result.unsqueeze(-1), dim=-1)
            acc = torch.sum(torch.sum(weights_truth, dim=-1) > 0)/weights_truth.shape[0]
            weights_truth[weights_truth > 1] = 1
            weights_loss = torch.nn.functional.binary_cross_entropy(weights.to(torch.float32), weights_truth.to(torch.float32))
            retval = {
                "loss": self.loss_param["lambda_diffusion"] * diffusion_loss + self.loss_param["lambda_crossentropy"] * weights_loss,
                "loss_diffusion": diffusion_loss,
                "loss_crossentropy": weights_loss,
                "acc": acc,
                "seen": this_batch_size
            }
        else:
            raise NotImplementedError()

        return retval

    def training_step(self, batch_data, batch_nb):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()
        retval = self.forward(batch_data, "train")
        self.manual_backward(retval["loss"])

        if self.optimizer_params["use_grad_norm_crip"]:
            total_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                self.model.parameters(), self.optimizer_params["grad_norm_clip_value"])
            retval["train_total_norm"] = total_norm
        opt.step()
        sch.step()
        self.training_step_outputs.append(retval)

        self.step+=1

        return retval

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        losses = losses_aggregation(outputs)
        for k, v in losses.items():
            self.log(k, v)

        self.training_step_outputs.clear()

    def validation_step(self, batch_data, batch_nb):
        retval = self.forward(batch_data, "val")

        self.validation_step_outputs.append(retval)
        return retval

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        losses = losses_aggregation(outputs)
        for k, v in losses.items():
            self.log("val_" + k, v)
        this_lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        print(this_lr)
        self.log("Lr", this_lr)
        self.validation_step_outputs.clear()

    def test_step(self, batch_data, batch_nb):
        retval = self.forward(batch_data, "test")

        self.test_step_outputs.append(retval)
        return retval

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        losses = losses_aggregation(outputs)
        for k, v in losses.items():
            self.log("test_" + k, v)
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        if self.optimizer_params["mode"] == "default":
            warmup_steps = self.optimizer_params["warmup_steps"]
            f = lambda t: min((t + 1) ** -0.5, (t + 1) * warmup_steps ** -1.5)
            lr = self.optimizer_params["constant"] * f(0)
        elif self.optimizer_params["mode"] == "cos_anneal":
            warmup_epoch = self.optimizer_params["warmup_epochs"]
            eta_min_linear = self.optimizer_params["eta_min_linear"]
            eta_min_cos = self.optimizer_params["eta_min_cos"]
            max_epoch = self.optimizer_params["max_epoch"]
            lr = self.optimizer_params["lr"]
            f = lambda t: max(t / (warmup_epoch), eta_min_linear / lr) if t < warmup_epoch else max(
                (1 + math.cos(math.pi * (t - warmup_epoch) / (max_epoch - warmup_epoch))) / 2, eta_min_cos / lr)
        elif self.optimizer_params["mode"] == "constant":
            warmup_epoch = self.optimizer_params["warmup_epochs"]
            eta_min_linear = self.optimizer_params["eta_min_linear"]
            lr = self.optimizer_params["lr"]
            f = lambda t: max(t / (warmup_epoch), eta_min_linear / lr) if t < warmup_epoch else 1.0
        else:
            raise NotImplementedError()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr,
                                     betas=(self.optimizer_params["Adam_beta1"], self.optimizer_params["Adam_beta2"]),
                                     eps=float(self.optimizer_params["Adam_eps"]))
        if self.optimizer_params["mode"] == "default":
            sch = lr_scheduler.LambdaLR(optimizer, lambda t: f(t) / f(0))
        else:
            sch = lr_scheduler.LambdaLR(optimizer, lambda t: f(t))
        return [[optimizer], [sch]]


def losses_aggregation(losses):
    agg_losses = {k: 0.0 for k in losses[0].keys()}
    for l in losses:
        print(l.items())
        for k, v in l.items():
            if (k == "seen"):
                agg_losses[k] += float(v)
            else:
                agg_losses[k] += float(v) * float(l["seen"])
    for k, v in agg_losses.items():
        if (k != "seen"):
            agg_losses[k] = agg_losses[k] / agg_losses["seen"]

    return agg_losses

