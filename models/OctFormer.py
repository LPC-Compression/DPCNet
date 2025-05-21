import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional, Any, Union, Callable
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.modules.transformer import _get_clones
import sys
sys.path.append("../")
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules import Linear, LayerNorm, Dropout, Conv1d
import torch.nn.functional as F

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class LeTransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 128, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LeTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)

        # Implementation of Conv Feedforward model
        self.linear1 = Linear(d_model, d_model, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.conv1 = Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.linear2 = Linear(d_model, d_model, **factory_kwargs)

    
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(LeTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_conv_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_conv_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block with 1d convolution
    def _ff_conv_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.activation(x)
        x = x.permute(0,2,1)
        x = self.linear2(x)
        x = self.dropout2(x)
    
        return x

# 1D conv
class PosEncodingBlock(nn.Module):
    def __init__(self, in_chans, embed_dim=128, s=1):
        super(PosEncodingBlock, self).__init__()
        self.proj = nn.Sequential(nn.Conv1d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, N)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class OctFormerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``False`` (disabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=False, d_model = 128):
        super(OctFormerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.pos_block = PosEncodingBlock(in_chans=d_model,embed_dim=d_model)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        convert_to_nested = False

        for index,mod in enumerate(self.layers):
            if convert_to_nested:
                output = mod(output, src_mask=mask)
            else:
                if(index == 0):
                    output = self.pos_block(output)
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output



class OctFormer(nn.Module):
    def __init__(self,sequence_size, dropout_rate, hidden = 128, nhead = 8, num_layer = 6, absolute_pos = "False", OctLeff = "True", OctPEG = "True"):
        print("OctFormer: sequence_size: {}, nhead: {}, num_layer: {}, hidden: {}, dropout: {}, absolute_pos: {}, OctLeff: {}, OctPEG: {}".format(
            sequence_size,
            nhead,
            num_layer,
            hidden,
            dropout_rate,
            absolute_pos,
            OctLeff,
            OctPEG
            ))
        super(OctFormer,self).__init__()
        self.embedding = nn.Linear(in_features=6,out_features=hidden)
        if(OctLeff == "True"):
            print("OctLeFF ✓")
            self.encoder_layer = LeTransformerEncoderLayer(d_model=hidden, dim_feedforward=hidden*4 ,nhead=nhead,batch_first=True)
        else:
            print("OctLeFF ✕")
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, dim_feedforward=hidden*4, nhead=nhead,batch_first=True)
        
        if(OctPEG == "True"):
            print("OctPEG ✓")
            self.transformerEncoder = OctFormerEncoder(self.encoder_layer,num_layers=num_layer,d_model=hidden)
        else:
            print("OctPEG ✕")
            self.transformerEncoder = nn.TransformerEncoder(self.encoder_layer,num_layers=num_layer)
        
        self.absolute_pos = False
        if(absolute_pos == "True"):
            print("absolute_pos ✓")
            self.absolute_pos = True
            self.pos_embedding = nn.Parameter(torch.randn(1, sequence_size, hidden))
        else:
            print("absolute_pos ✕")
        
        self.MLP1 = nn.Linear(in_features=hidden,out_features=hidden)
        self.MLP2 = nn.Linear(in_features=hidden,out_features=256)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, features):
        out = self.embedding(features)
        b,n,_ = out.shape
        if(self.absolute_pos == True):
            out += self.pos_embedding[:, :(n)]
        out = self.transformerEncoder(out)
        out = self.MLP1(out)
        out = self.dropout(out)
        out = self.MLP2(out)
        return out

if __name__ == '__main__':
    model = OctFormer(sequence_size=128,dropout_rate = 0.5,absolute_pos="True", OctLeff="True", OctPEG="True", hidden=256, nhead=8, num_layer=6)
    h = torch.zeros((256,16,6)) # batch_size =  256, sequence = 16, dimention=6
    print(model(h).shape)
    # torch.save(model,"1.pth")
        
