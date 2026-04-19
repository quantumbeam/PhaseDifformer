import torch
from torch import Tensor
from torch.nn.modules import Module
from torch.nn import Linear, Dropout, LayerNorm, MultiheadAttention
from typing import Optional, Any, Union, Callable
from torch.nn import ModuleList
from torch.nn import functional as F
import copy
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout




def mlp(channels, use_bn=True, use_dropout=True, drop_prob=0.3):
    net = [([Lin(channels[i - 1], channels[i])], channels[i]) for i in range(1, len(channels))]
    net = [(n + [ReLU()], c) for n, c in net[:-1]] + [net[-1]]

    if use_bn:
        net = [(n + [BN(c, eps=1e-10)], c) for n, c in net[:-1]] + [net[-1]]
    if use_dropout:
        net = [(n + [Dropout(drop_prob)], c) for n, c in net[:-1]] + [net[-1]]
    net = [Seq(*n) for n, _ in net]
    net = Seq(*net)
    return net


class TransformerEncoderLayer_SelfAttention(Module):
    r"""This implementation is https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer_SelfAttention, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayer_SelfAttention, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoderLayer_CrossAttention(Module):
    r"""This implementation is modified from https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, given_weights: bool = False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer_CrossAttention, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

        if given_weights:
            self.nhead = nhead
            self.dim_per_head = d_model // nhead
            self.mlp = mlp([self.dim_per_head, self.dim_per_head], use_bn=False, use_dropout=False)

        self.given_weights = given_weights
    def __setstate__(self, state):
        super(TransformerEncoderLayer_CrossAttention, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src_1: Tensor, src_2: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            attn_weights: Weights for cross attention (optional).

        Shape:
            attn_weights: [batch, nhead, src_1_len, src_2_len]
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src_2.dtype
        )

        if self.norm_first:
            x = src_1 + self._ca_block(self.norm1(src_1), self.norm2(src_2), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))

        else:
            x = self.norm1(src_1 + self._ca_block(src_1, src_2, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _ca_block(self, x_1: Tensor, x_2: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x_1, x_2, x_2,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)


    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderSelf_MultiLayer(Module):
    def __init__(self, layer_num: int, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False) -> None:
        super(TransformerEncoderSelf_MultiLayer, self).__init__()

        self.xrd_transformer_encoder = torch.nn.Sequential()
        for i in range(layer_num):
            tmp = TransformerEncoderLayer_SelfAttention(d_model, nhead,
                                                        dim_feedforward, dropout,
                                                        activation, layer_norm_eps, batch_first, norm_first)
            self.xrd_transformer_encoder.add_module("xrd_transformer_encoder_self" + str(i), tmp)

    def forward(self, x, mask = None):
        for i in range(len(self.xrd_transformer_encoder)):
            x = self.xrd_transformer_encoder[i](x, src_key_padding_mask = mask)

        return x

class TransformerEncoderLayer_SelfAndCrossAttention(Module):
    def __init__(self, layer_order_for_src1: list, layer_order_for_src2: list, d_model:int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        super(TransformerEncoderLayer_SelfAndCrossAttention, self).__init__()
        self.d_model = d_model

        self.layer_order_for_src1 = layer_order_for_src1
        self.layer_order_for_src2 = layer_order_for_src2

        self.transformer_encoder_for_src1 = torch.nn.Sequential()
        self.transformer_encoder_for_src2 = torch.nn.Sequential()

        for i in range(len(layer_order_for_src1)):
            if layer_order_for_src1[i] == "s":
                tmp = TransformerEncoderLayer_SelfAttention(
                    d_model, nhead, dim_feedforward, dropout,
                    activation, layer_norm_eps, batch_first,  norm_first, device, dtype
                )
                self.transformer_encoder_for_src1.add_module("transformer_encoder_for_src1_self" + str(i), tmp)
            elif layer_order_for_src1[i] == "c":
                tmp = TransformerEncoderLayer_CrossAttention(
                    d_model, nhead, dim_feedforward, dropout,
                    activation, layer_norm_eps, batch_first, norm_first, device, dtype
                )
                self.transformer_encoder_for_src1.add_module("transformer_encoder_for_src1_cross" + str(i), tmp)
            elif layer_order_for_src1[i] == "n":
                pass
            else:
                raise NotImplementedError()

            if layer_order_for_src2[i] == "s":
                tmp = TransformerEncoderLayer_SelfAttention(
                    d_model, nhead, dim_feedforward, dropout,
                    activation, layer_norm_eps, batch_first,  norm_first, device, dtype
                )
                self.transformer_encoder_for_src2.add_module("transformer_encoder_for_src2_self" + str(i), tmp)
            elif layer_order_for_src2[i] == "c":
                tmp = TransformerEncoderLayer_CrossAttention(
                    d_model, nhead, dim_feedforward, dropout,
                    activation, layer_norm_eps, batch_first, norm_first, device, dtype
                )
                self.transformer_encoder_for_src2.add_module("transformer_encoder_for_src2_cross" + str(i), tmp)
            elif layer_order_for_src2[i] == "n":
                pass
            else:
                raise NotImplementedError()

    def forward(self, src1: Tensor, src2: Tensor, src1_key_padding_mask:Tensor, src2_key_padding_mask:Tensor):
        count_src1 = 0
        count_src2 = 0
        for i in range(len(self.layer_order_for_src1)):
            if self.layer_order_for_src1[i] == "s":
                src1 = self.transformer_encoder_for_src1[count_src1](src1, src_key_padding_mask=src1_key_padding_mask)
                count_src1 += 1
            elif self.layer_order_for_src1[i] == "c":
                src1 = self.transformer_encoder_for_src1[count_src1](src1, src2, src_key_padding_mask=src2_key_padding_mask)
                count_src1 += 1
            else:
                pass

            if self.layer_order_for_src2[i] == "s":
                src2 = self.transformer_encoder_for_src2[count_src2](src2, src_key_padding_mask=src2_key_padding_mask)
                count_src2 += 1
            elif self.layer_order_for_src2[i] == "c":
                src2 = self.transformer_encoder_for_src2[count_src1](src2, src1, src_key_padding_mask=src1_key_padding_mask)
                count_src2 += 1
            else:
                pass
        return src1, src2


