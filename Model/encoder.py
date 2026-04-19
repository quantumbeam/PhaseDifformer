import torch
from torch import nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.data import Data, Batch
from . import utils
import math
torch.manual_seed(42)

class Encoder(torch.nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
            layer_norm_eps,
            positional_encoding_channels,
            positional_encoding_use_bn,
            positional_encoding_use_dropout,
            positional_encoding_drop_prob,
            intensity_encoding_channels,
            intensity_encoding_use_bn,
            intensity_encoding_use_dropout,
            intensity_encoding_drop_prob,
    ):
        super(Encoder, self).__init__()
        if num_layers != 0:
            self.encoder = utils.TransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                mlp_channels=None,
                mlp_use_bn=None,
                mlp_use_dropout=None,
                mlp_drop_prob=None,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                non_mlp=True
            )
        else:
            self.encoder = None

        self.position_patch_encoding = utils.PosIntEncoding(
            positional_encoding_channels=positional_encoding_channels,
            positional_encoding_use_bn=positional_encoding_use_bn,
            positional_encoding_use_dropout=positional_encoding_use_dropout,
            positional_encoding_drop_prob=positional_encoding_drop_prob,
            intensity_encoding_channels=intensity_encoding_channels,
            intensity_encoding_use_bn=intensity_encoding_use_bn,
            intensity_encoding_use_dropout=intensity_encoding_use_dropout,
            intensity_encoding_drop_prob=intensity_encoding_drop_prob
        )

    def forward(self, angle, intensity, mask, add_embedding=None):
        embedding = self.position_patch_encoding(angle, intensity)
        if add_embedding is not None:
            add_dim = add_embedding.shape[1]
            embedding = torch.concat([add_embedding, embedding], dim=1)
            mask = torch.concat([torch.zeros((embedding.shape[0], add_dim),dtype=torch.bool, device=mask.device), mask], dim=-1)
        else:
            add_dim=0

        if self.encoder is not None:
            return self.encoder(embedding, src_key_padding_mask=mask)[:, add_dim:, :]
        else:
            return embedding[:, add_dim:, :]
class Encoder_w_cls(torch.nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
            layer_norm_eps,
            positional_encoding_channels,
            positional_encoding_use_bn,
            positional_encoding_use_dropout,
            positional_encoding_drop_prob,
            intensity_encoding_channels,
            intensity_encoding_use_bn,
            intensity_encoding_use_dropout,
            intensity_encoding_drop_prob,
    ):
        super(Encoder_w_cls, self).__init__()
        self.encoder = utils.TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            mlp_channels=None,
            mlp_use_bn=None,
            mlp_use_dropout=None,
            mlp_drop_prob=None,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            non_mlp=True
        )

        self.position_patch_encoding = utils.PosIntEncoding(
            positional_encoding_channels=positional_encoding_channels,
            positional_encoding_use_bn=positional_encoding_use_bn,
            positional_encoding_use_dropout=positional_encoding_use_dropout,
            positional_encoding_drop_prob=positional_encoding_drop_prob,
            intensity_encoding_channels=intensity_encoding_channels,
            intensity_encoding_use_bn=intensity_encoding_use_bn,
            intensity_encoding_use_dropout=intensity_encoding_use_dropout,
            intensity_encoding_drop_prob=intensity_encoding_drop_prob,
        )

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model)/math.sqrt(d_model))

    def forward(self, angle, intensity, mask, add_embedding=None, all_return=False):
        embedding = self.position_patch_encoding(angle, intensity)
        if add_embedding is not None:
            add_dim = add_embedding.shape[1]
            embedding = torch.concat([add_embedding, embedding], dim=1)
            mask = torch.concat([torch.zeros((embedding.shape[0], add_dim),dtype=torch.bool, device=mask.device), mask], dim=-1)
        else:
            add_dim=0
        embedding = torch.concatenate([self.cls_token.repeat(embedding.shape[0], 1, 1), embedding], dim=1)
        mask = torch.concat([torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device), mask], dim=1)
        if all_return:
            return self.encoder(embedding, src_key_padding_mask=mask)
        else:
            return self.encoder(embedding, src_key_padding_mask=mask)[:, 0, :]
