import torch
from . import encoder
from . import utils

class Decoder(torch.nn.Module):
    def __init__(
            self,
            mlp_channels,
            mlp_use_bn,
            mlp_use_dropout,
            mlp_drop_prob
    ):
        super(Decoder, self).__init__()
        self.mlp = utils.MLP(
            mlp_channels,
            mlp_use_bn,
            mlp_use_dropout,
            mlp_drop_prob
        )

    def forward(self, embedding):
        embedding = self.mlp(embedding)
        return embedding[:, :, 0]

