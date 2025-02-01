# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
import pdb
from torch import nn

from ..transformer_decoder.maskformer_transformer_decoder import (
    MultiScaleMaskedTransformerDecoder,
)
from ..pixel_decoder.fpn import build_pixel_decoder


class MaskFormerHead(nn.Module):
    def __init__(
        self,
        cfg,
        input_shape: Dict,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict 后续如果加类别再说
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight 后续如果加类别再说
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = {
            k: v for k, v in input_shape.items() if k in cfg.SEM_SEG_HEAD.IN_FEATURES
        }
        feature_stride = [v["strides"] for k, v in input_shape.items()]
        self.pixel_decoder = build_pixel_decoder(cfg, input_shape)
        self.predictor = MultiScaleMaskedTransformerDecoder(
            cfg,
            input_shape=input_shape,
            feature_size=[int(i / max(feature_stride)) for i in cfg.INPUT_SIZE],
        )

    def forward(self, features, feature_dict):
        return self.layers(features, feature_dict)

    def layers(self, noised_features, feature_dict):
        # features is the output dict of backbone
        _, _, multi_scale_features = self.pixel_decoder.forward_features(
            noised_features
        )
        predictions = self.predictor(multi_scale_features, feature_dict)
        return predictions
