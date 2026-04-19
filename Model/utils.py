import copy
import pickle
import torch
from torch import nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from tqdm import tqdm
from Model import model
import yaml
torch.manual_seed(42)

def MLP(channels, use_bn=True, use_dropout=True, drop_prob=0.3):
    net = [([Lin(channels[i - 1], channels[i])], channels[i]) for i in range(1, len(channels))]
    net = [(n + [ReLU()], c) for n, c in net[:-1]] + [net[-1]]

    if use_bn:
        net = [(n + [BN(c)], c) for n, c in net[:-1]] + [net[-1]]
    if use_dropout:
        net = [(n + [Dropout(drop_prob)], c) for n, c in net[:-1]] + [net[-1]]
    net = [Seq(*n) for n, _ in net]
    net = Seq(*net)
    return net

def load_pkl(path_pkl):
    with open(path_pkl, mode='br') as fi:
        data = pickle.load(fi)
    return data

def padding_from_batch(batch, name, generate_mask=False):
    data_list = [data[name] for data in batch.to_data_list()]
    padded_batch = nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=0)

    if generate_mask:
        original_lengths = torch.tensor([data.shape[0] for data in data_list], device=padded_batch.device)
        max_length = padded_batch.size(1)
        mask = torch.arange(max_length, device=padded_batch.device).expand(len(data_list), max_length) >= original_lengths.unsqueeze(1)
    else:
        mask = None

    return padded_batch, mask

import time
from collections import defaultdict
def padding_from_batch_multi(batch, names, generate_mask=False):
    data_list_all = batch.to_data_list()
    result = {}

    for i, name in enumerate(names):
        data_list = [data[name] for data in data_list_all]
        padded_batch = nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=0)

        if generate_mask and i == 0:
            original_lengths = torch.tensor(
                [data.shape[0] for data in data_list], device=padded_batch.device
            )
            max_length = padded_batch.size(1)
            mask = torch.arange(max_length, device=padded_batch.device).expand(
                len(data_list), max_length) >= original_lengths.unsqueeze(1)
        else:
            mask = None

        result[name] = (padded_batch, mask)

    return result

def padding_from_list(data_list, generate_mask=False):
    padded_batch = nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=0)

    if generate_mask:
        original_lengths = torch.tensor([data.shape[0] for data in data_list], device=padded_batch.device)
        max_length = padded_batch.size(1)
        mask = torch.arange(max_length, device=padded_batch.device).expand(len(data_list), max_length) >= original_lengths.unsqueeze(1)
    else:
        mask = None

    return padded_batch, mask


class TransformerEncoder(torch.nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            num_layers,
            mlp_channels,
            mlp_use_bn = True,
            mlp_use_dropout = True,
            mlp_drop_prob = 0.3,
            dim_feedforward = 2048,
            dropout = 0.1,
            layer_norm_eps = 1e-5,
            non_mlp = False
    ):
        super(TransformerEncoder, self).__init__()
        assert d_model % nhead == 0, "Encoder initialization error. d_model should be divided by nhead."
        self.nhead = nhead
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, dim_feedforward = dim_feedforward, dropout = dropout, layer_norm_eps = layer_norm_eps, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.non_mlp = non_mlp
        if not non_mlp:
            self.MLP = MLP(mlp_channels, use_bn=mlp_use_bn, use_dropout=mlp_use_dropout, drop_prob=mlp_drop_prob)
            assert d_model == mlp_channels[0], "d_model and MLP First channel should be same."


    def forward(self, input, src_key_padding_mask):
        retval_transformer = self.transformer_encoder(input.to(torch.float32), src_key_padding_mask=src_key_padding_mask)

        if not self.non_mlp:
            head_val = retval_transformer[:, 0, :]
            return self.MLP(head_val)
        else:
            return retval_transformer



class PositionalEncoding(torch.nn.Module):
    def __init__(
            self,
            positional_encoding_channels,
            positional_encoding_use_bn,
            positional_encoding_use_dropout,
            positional_encoding_drop_prob
    ):
        super(PositionalEncoding, self).__init__()

        self.PositionalEncoding = MLP(positional_encoding_channels, positional_encoding_use_bn, positional_encoding_use_dropout, positional_encoding_drop_prob)

    def forward(self, position):
        embedding = self.PositionalEncoding(position.to(torch.float32))

        return embedding.to(torch.float32)


class PosIntEncoding(torch.nn.Module):
    def __init__(
            self,
            positional_encoding_channels,
            positional_encoding_use_bn,
            positional_encoding_use_dropout,
            positional_encoding_drop_prob,
            intensity_encoding_channels,
            intensity_encoding_use_bn,
            intensity_encoding_use_dropout,
            intensity_encoding_drop_prob,
    ):
        super(PosIntEncoding, self).__init__()
        self.angle_encoding = PositionalEncoding(
            positional_encoding_channels,
            positional_encoding_use_bn,
            positional_encoding_use_dropout,
            positional_encoding_drop_prob
        )
        self.intensity_encoding = PositionalEncoding(
            intensity_encoding_channels,
            intensity_encoding_use_bn,
            intensity_encoding_use_dropout,
            intensity_encoding_drop_prob
        )

    def forward(self, angle, intensity):
        position_embedding = self.angle_encoding(angle.unsqueeze(-1))
        intensity_embedding = self.intensity_encoding(intensity.unsqueeze(-1))

        embedding = position_embedding + intensity_embedding

        return embedding.to(torch.float32)

class DDPM(nn.Module):
    def __init__(
            self,
            encoder_param_mixed,
            encoder_param_noisy,
            diffusion_param
    ):
        super().__init__()
        if diffusion_param["method"] == "linear":
            self.betas = torch.linspace(diffusion_param["beta_min"], diffusion_param["beta_max"],
                                        steps=diffusion_param["t_max"])
        else:
            raise NotImplementedError()
        self.times = torch.linspace(-1, 1, diffusion_param["t_max"])
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.t_max = diffusion_param["t_max"]

        print("alpha_bars: " + str(self.alpha_bars[0]) + "~" + str(self.alpha_bars[-1]))

    def diffusion_process(self, x0, t=None):
        if t is None:
            t = torch.randint(low=1, high=self.t_max, size=(x0.shape[0],))
        else:
            t = t.to(torch.int32)
        noise = torch.randn_like(x0, device=x0.device)
        alpha_bar = self.alpha_bars[t].reshape(-1, 1).to(noise.device)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

        return xt, self.times[t.to(torch.int32).to(self.times.device)].to(noise.device), noise

    def denoising_process(
            self, model, angle, intensity_input, intensity_noisy, mask, ts, device,
            search_result_index=None, database_name=None,
            confidence=None):
        batch_size = angle.shape[0]
        intensity_input = intensity_input.to(device)
        intensity_noisy = intensity_noisy.to(device)

        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for t in tqdm(reversed(range(1, ts.to(torch.int32)))):
                time_tensor = (torch.ones(batch_size, device=device) * t)
                time_tensor = self.times[time_tensor.to(torch.int32).to(self.times.device)].to(device)
                prediction_noise = model.model_dif_transformer(
                    angle, intensity_input, intensity_noisy,
                    time_tensor, mask,
                    search_result_index.to(device), database_name,
                    confidence=confidence.to(device)
                )[0]

                intensity_noisy = self._calc_denoising_one_step(intensity_noisy, time_tensor, prediction_noise)

        return intensity_noisy

    def _calc_denoising_one_step(self, pattern_noisy, time_tensor, prediction_noise):
        beta = self.betas[time_tensor.to(self.betas.device).to(torch.int32)].reshape(-1, 1).to(pattern_noisy.device)
        sqrt_alpha = torch.sqrt(self.alphas[time_tensor.to(self.alphas.device).to(torch.int32)].reshape(-1, 1)).to(pattern_noisy.device)
        alpha_bar = self.alpha_bars[time_tensor.to(self.alpha_bars.device).to(torch.int32)].reshape(-1, 1).to(pattern_noisy.device)
        sigma_t = torch.sqrt(beta)
        noise = torch.randn_like(pattern_noisy, device=pattern_noisy.device) \
            if time_tensor[0].item() > 1 else torch.zeros_like(pattern_noisy, device=pattern_noisy.device)
        pattern_noisy = 1 / sqrt_alpha * (pattern_noisy - (beta / (torch.sqrt(1 - alpha_bar))) * prediction_noise) + sigma_t * noise
        return pattern_noisy


def select_indices(batch, rate):
    if rate == 1.0:
        return torch.arange(batch.shape[0], device=batch.device)
    mask = torch.rand(batch.shape[0], device=batch.device) < rate
    return torch.nonzero(mask, as_tuple=False).flatten()

def get_batch_detected(angle):
    resets = (angle[1:] < angle[:-1]).to(torch.int64)
    resets = torch.cat([torch.zeros(1, dtype=torch.int64, device=angle.device), resets])
    return torch.cumsum(resets, dim=0)


def load_model(path_config, path_model, path_database=None):
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
    if path_database is None:
        path_database = cfg["path_database"]

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

    modelptl = best_model
    modelptl.model.eval()

    return modelptl.model.eval(), cfg

def get_index_given_rate(source_index, search_correct_rate, num_use, search_max_num):
    source_index_copy = copy.deepcopy(source_index)
    search_result = torch.randint(
        low=0, high=search_max_num, size=(source_index.shape[0], source_index.shape[1]),
        device=source_index.device).to(torch.float32)
    random_tmp = torch.rand_like(source_index_copy, device=source_index.device)
    source_index_copy[torch.logical_or(random_tmp > search_correct_rate, source_index_copy < 0)] = \
        search_result[torch.logical_or(random_tmp > search_correct_rate, source_index_copy < 0)]
    search_result = torch.randint(
        low=0, high=search_max_num, size=(source_index.shape[0], num_use),
        device=source_index.device).to(torch.float32)
    search_result[:, :source_index.shape[1]] = source_index_copy

    return search_result
