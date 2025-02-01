# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, Optional, Union

import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from torch import nn
from torch.nn import functional as F

from ..transformer_decoder.modules.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.modules.transformer import TransformerEncoder, TransformerEncoderLayer


def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    input_shape = {
        k: v for k, v in input_shape.items() if k in cfg.SEM_SEG_HEAD.IN_FEATURES
    }
    if name == "TransformerEncoderPixelDecoder":
        model = TransformerEncoderPixelDecoder(cfg, input_shape)
    else:
        model = BasePixelDecoder(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


# This is a modified FPN decoder.
class BasePixelDecoder(nn.Module):
    def __init__(
            self,
            cfg,
            input_shape: Dict[str, ShapeSpec],
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            norm (str or callable): normalization for all conv layers
        """

        super(BasePixelDecoder, self).__init__()
        conv_dim = cfg.SEM_SEG_HEAD.CONV_DIM
        norm = cfg.SEM_SEG_HEAD.NORM
        avg_pool = cfg.BACKBONE.EFFICIENTNET.FEATURE_POOL_SIZE != 0

        input_shape = sorted(input_shape.items(), key=lambda x: x[1]["strides"])  # input_shape是 backbone.output_shape()
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v["channels"] for k, v in input_shape]
        if avg_pool:
            feature_channels = [feature_channel * 2 for feature_channel in feature_channels]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)

                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )

                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)

            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, stride=1, bias=use_bias, norm=lateral_norm, padding="same",
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )

                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.maskformer_num_feature_levels = 3  # always use 3 scales, 4 features

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            # up_conv = self.up_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="bilinear")
                y = output_conv(y)
            if num_cur_levels <= self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return y, None, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)


class TransformerEncoderOnly(nn.Module):
    def __init__(
            self,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, pos=pos_embed, mask=mask)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


# This is a modified FPN decoder with extra Transformer encoder that processes the lowest-resolution feature map.
class TransformerEncoderPixelDecoder(BasePixelDecoder):
    def __init__(
            self,
            cfg,
            input_shape: Dict[str, ShapeSpec],
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            norm (str or callable): normalization for all conv layers
        """
        transformer_dropout = cfg.SEM_SEG_HEAD.DROPOUT
        transformer_nheads = cfg.SEM_SEG_HEAD.NHEADS
        transformer_dim_feedforward = cfg.SEM_SEG_HEAD.DIM_FEEDFORWARD
        transformer_enc_layers = cfg.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        transformer_pre_norm = cfg.SEM_SEG_HEAD.PRE_NORM
        self.encoder_mask = cfg.SEM_SEG_HEAD.ENCODER_MASK
        avg_pool = cfg.BACKBONE.EFFICIENTNET.FEATURE_POOL_SIZE != 0
        norm: Optional[Union[str, Callable]] = None

        super().__init__(cfg, input_shape)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1]["strides"])
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v["channels"] for k, v in input_shape]
        if avg_pool:
            feature_channels = [feature_channel * 2 for feature_channel in feature_channels]
        conv_dim = cfg.SEM_SEG_HEAD.CONV_DIM

        in_channels = feature_channels[len(self.in_features) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)
        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # update layer
        use_bias = norm == ""
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    def generate_mask(self, feature_size, neighbor_size, ):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = 1 - mask.view(h * w, h * w)
        return mask.bool().cuda()

    def forward_features(self, features_dict):
        if self.encoder_mask:
            feature_size = features_dict["res4"].shape[-2:]
            mask = self.generate_mask(feature_size, (7, 7))
        else:
            mask = None
        # 记得传入有噪音的那个feature dict
        multi_scale_features = []
        transformer_encoder_features = None
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features_dict[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, mask, pos)
                y = output_conv(transformer)
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="bilinear")
                # y = cur_fpn  # for ablation
                y = output_conv(y)
            if num_cur_levels <= self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return y, transformer_encoder_features, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)
