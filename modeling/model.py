# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn

from .backbone import build_backbone
from .meta_arch import build_seg_head


class UniAS(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.backbone = build_backbone(cfg)
        if not cfg.BACKBONE.TRAINABLE:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.sem_seg_head = build_seg_head(cfg, self.backbone.output_shape())
        self.num_queries = cfg.MASK_FORMER.NUM_OBJECT_QUERIES
        self.size_divisibility = cfg.MASK_FORMER.SIZE_DIVISIBILITY
        input_shape = {
            k: v
            for k, v in self.backbone.output_shape().items()
            if k in cfg.SEM_SEG_HEAD.IN_FEATURES
        }
        input_shape = sorted(
            input_shape.items(), key=lambda x: x[1]["strides"]
        )  # input_shape是 backbone.output_shape()
        self.strides = [v["strides"] for k, v in input_shape]

    def cuda(self):
        self.device = torch.device("cuda")
        return super(UniAS, self).cuda()

    def cpu(self):
        self.device = torch.device("cpu")
        return super(UniAS, self).cpu()

    def forward(self, batched_inputs, training=True):
        if not isinstance(batched_inputs, dict):
            clsidx = 0
            features, noised_features = self.backbone(batched_inputs, clsidx, training)
            outputs = self.sem_seg_head(noised_features, features)
            return outputs
        shape = batched_inputs["image"].shape[-2:]
        if self.size_divisibility:
            assert (
                shape[0] % max(self.strides) == 0 and shape[1] % max(self.strides) == 0
            ), "size_divisibility on, but img size is not divisible by stride"

        if batched_inputs["image"].device != self.device:
            input_data = batched_inputs["image"].to(self.device)
        else:
            input_data = batched_inputs["image"]

        clsidx = batched_inputs["clsidx"]
        features, noised_features = self.backbone(input_data, clsidx, training)
        outputs = self.sem_seg_head(noised_features, features)
        return outputs

    @classmethod
    def check_requires_grad(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Parameter {name} requires gradient.")
            else:
                print(f"Parameter {name} does not require gradient.")
