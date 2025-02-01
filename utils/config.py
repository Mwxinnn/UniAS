# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the datasets mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in datasets mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = 32

    # TRAINER config
    # weight decay on embedding
    cfg.TRAINER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.TRAINER.OPTIMIZER = "ADAMW"
    cfg.TRAINER.BACKBONE_MULTIPLIER = 0.1

    # loss
    cfg.LOSS.IMAGE_CLS_WEIGHT = 1
    cfg.LOSS.FEATURE_CLS_WEIGHT = 1
    cfg.LOSS.BALANCE_WEIGHT = 1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()
    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = False
    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res4"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "TransformerEncoderPixelDecoder"

    # EFFICIENTNET backbone
    cfg.MODEL.EFFICIENTNET = CN()
    cfg.MODEL.EFFICIENTNET.LAYERS = [1, 2, 3, 4]
    cfg.MODEL.EFFICIENTNET.FEATURE_JITTER_SCALE = 20
    cfg.MODEL.EFFICIENTNET.FEATURE_JITTER_PROB = 1

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 288
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
    cfg.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
