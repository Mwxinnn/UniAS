# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/transformer.py
"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import pdb
from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class SpaticalNChannelAttn(nn.Module):
    def __init__(self,
                 spatial_hdim: int,
                 channel_hdim: int,
                 dropout: float,
                 pre_norm: bool,
                 num_heads: int,
                 dim_feedforward: int,
                 cross_first: bool,
                 **kwargs
                 ):
        super().__init__()
        self.SpatialSelfAttnLayer = SelfAttentionLayer(
            d_model=spatial_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        self.ChannelSelfAttnLayer = SelfAttentionLayer(
            d_model=channel_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )

        self.SpatialCrossAttnLayer = CrossAttentionLayer(
            d_model=spatial_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        self.ChannelCrossAttnLayer = CrossAttentionLayer(
            d_model=channel_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )

        self.SpatialFFNLayer = FFNLayer(
            d_model=spatial_hdim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        self.ChannelFFNLayer = FFNLayer(
            d_model=channel_hdim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        self.cross_first = cross_first

    #TODO
    def forward(self, output, src, crossattn_mask, selfattn_mask, pos, spatial_query_pos):
        if self.cross_first:
            spatial_output = self.SpatialCrossAttnLayer(
                output, src,
                memory_mask=crossattn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos, query_pos=spatial_query_pos
            )
            spatial_output = self.SpatialSelfAttnLayer(
                spatial_output, tgt_mask=selfattn_mask,
                tgt_key_padding_mask=None,
                query_pos=spatial_query_pos
            )
            spatial_output = self.SpatialFFNLayer(spatial_output)
            # ========================================================
            channel_output = self.ChannelCrossAttnLayer(
                output.permute((2, 1, 0)), src.permute((2, 1, 0)),
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos.permute((2, 1, 0))
            )
            channel_output = self.ChannelSelfAttnLayer(
                channel_output, tgt_mask=None,
                tgt_key_padding_mask=None,
            )
            channel_output = self.ChannelFFNLayer(channel_output)
            output = spatial_output + channel_output.permute((2, 1, 0))
            return output
        else:
            spatial_output = self.SpatialSelfAttnLayer(
                output, tgt_mask=selfattn_mask,
                tgt_key_padding_mask=None,
                query_pos=spatial_query_pos
            )
            spatial_output = self.SpatialCrossAttnLayer(
                spatial_output, src,
                memory_mask=crossattn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos, query_pos=spatial_query_pos
            )
            spatial_output = self.SpatialFFNLayer(spatial_output)
            # ========================================================
            channel_output = self.ChannelSelfAttnLayer(
                output.permute((2, 1, 0)), tgt_mask=None,
                tgt_key_padding_mask=None,
            )
            channel_output = self.ChannelCrossAttnLayer(
                channel_output, src.permute((2, 1, 0)),
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos.permute((2, 1, 0))
            )
            channel_output = self.ChannelFFNLayer(channel_output)
            output = spatial_output + channel_output.permute((2, 1, 0))
            return output


class SpatialAttn(nn.Module):
    def __init__(self,
                 spatial_hdim: int,
                 dropout: float,
                 pre_norm: bool,
                 num_heads: int,
                 dim_feedforward: int,
                 cross_first: bool,
                 **kwargs
                 ):
        super().__init__()
        self.SpatialSelfAttnLayer = SelfAttentionLayer(
            d_model=spatial_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        self.SpatialCrossAttnLayer = CrossAttentionLayer(
            d_model=spatial_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        self.SpatialFFNLayer = FFNLayer(
            d_model=spatial_hdim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        self.cross_first = cross_first

    def forward(self, output, src, crossattn_mask, selfattn_mask, pos, spatial_query_pos, channel_query_pos):
        if self.cross_first:
            spatial_output = self.SpatialCrossAttnLayer(
                output, src,
                memory_mask=crossattn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos, query_pos=spatial_query_pos
            )
            spatial_output = self.SpatialSelfAttnLayer(
                spatial_output, tgt_mask=selfattn_mask,
                tgt_key_padding_mask=None,
                query_pos=spatial_query_pos
            )
            spatial_output = self.SpatialFFNLayer(spatial_output)
            return spatial_output
        else:
            spatial_output = self.SpatialSelfAttnLayer(
                output, tgt_mask=selfattn_mask,
                tgt_key_padding_mask=None,
                query_pos=spatial_query_pos
            )
            spatial_output = self.SpatialCrossAttnLayer(
                spatial_output, src,
                memory_mask=crossattn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos, query_pos=spatial_query_pos
            )
            spatial_output = self.SpatialFFNLayer(spatial_output)
            return spatial_output


class SelfSpatialNChannelAttn(nn.Module):
    def __init__(self,
                 spatial_hdim: int,
                 channel_hdim: int,
                 dropout: float,
                 pre_norm: bool,
                 num_heads: int,
                 dim_feedforward: int,
                 **kwargs
                 ):
        super().__init__()
        self.SpatialSelfAttnLayer1 = SelfAttentionLayer(
            d_model=spatial_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        self.ChannelSelfAttnLayer1 = SelfAttentionLayer(
            d_model=channel_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )

        self.SpatialSelfAttnLayer2 = SelfAttentionLayer(
            d_model=spatial_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        self.ChannelSelfAttnLayer2 = SelfAttentionLayer(
            d_model=channel_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )

        self.SpatialFFNLayer = FFNLayer(
            d_model=spatial_hdim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        self.ChannelFFNLayer = FFNLayer(
            d_model=channel_hdim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            normalize_before=pre_norm,
        )

    def forward(self, output, crossattn_mask, selfattn_mask, spatial_query_pos, channel_query_pos):
        spatial_output = self.SpatialSelfAttnLayer1(
            output, tgt_mask=crossattn_mask,
            tgt_key_padding_mask=None,
            query_pos=spatial_query_pos
        )
        spatial_output = self.SpatialSelfAttnLayer2(
            spatial_output, tgt_mask=selfattn_mask,
            tgt_key_padding_mask=None,
            query_pos=spatial_query_pos
        )
        spatial_output = self.SpatialFFNLayer(spatial_output)
        # ========================================================
        channel_output = self.ChannelSelfAttnLayer1(
            output.permute((2, 1, 0)), tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=channel_query_pos.permute((2, 1, 0))
        )
        channel_output = self.ChannelSelfAttnLayer2(
            channel_output, tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=channel_query_pos.permute((2, 1, 0))
        )
        channel_output = self.ChannelFFNLayer(channel_output)
        output = spatial_output + channel_output.permute((2, 1, 0))
        return output


class SelfSpatialAttn(nn.Module):
    def __init__(self,
                 spatial_hdim: int,
                 channel_hdim: int,
                 dropout: float,
                 pre_norm: bool,
                 num_heads: int,
                 dim_feedforward: int,
                 **kwargs
                 ):
        super().__init__()
        self.SpatialSelfAttnLayer1 = SelfAttentionLayer(
            d_model=spatial_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )
        self.SpatialSelfAttnLayer2 = SelfAttentionLayer(
            d_model=spatial_hdim,
            nhead=num_heads,
            dropout=dropout,
            normalize_before=pre_norm,
        )

        self.SpatialFFNLayer = FFNLayer(
            d_model=spatial_hdim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            normalize_before=pre_norm,
        )

    def forward(self, output, crossattn_mask, selfattn_mask, spatial_query_pos, channel_query_pos):
        spatial_output = self.SpatialSelfAttnLayer1(
            output, tgt_mask=crossattn_mask,
            tgt_key_padding_mask=None,
            query_pos=spatial_query_pos
        )
        spatial_output = self.SpatialSelfAttnLayer2(
            spatial_output, tgt_mask=selfattn_mask,
            tgt_key_padding_mask=None,
            query_pos=spatial_query_pos
        )
        spatial_output = self.SpatialFFNLayer(spatial_output)
        return spatial_output


def get_transformer_block(name, **cfg):
    if name == "SpaticalNChannelAttn":
        return SpaticalNChannelAttn(**cfg)
    elif name == "SpatialAttn":
        return SpatialAttn(**cfg)
    else:
        raise NotImplementedError


def get_ablation_transformer_block(name, **cfg):
    if name == "SelfSpatialNChannelAttn":
        return SelfSpatialNChannelAttn(**cfg)
    elif name == "SelfSpatialAttn":
        return SelfSpatialAttn(**cfg)
    else:
        raise NotImplementedError


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
