"""__init__.py - all efficientnet models.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).


__version__ = "0.7.1"

from .model import EfficientNet
from .utils import (
    get_model_params,
    get_outparams
)


class EfficientnetB6(EfficientNet):
    def __init__(self, cfg, pretrained, pretrained_model="", **override_params):
        layers = cfg.BACKBONE.EFFICIENTNET.LAYERS
        outblocks, outplanes, outstrides = get_outparams("efficientnet_b6", layers)
        blocks_args, global_params = get_model_params("efficientnet_b6", override_params)
        feature_jitter_scale = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_SCALE
        feature_jitter_prob = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_PROB

        feature_avg_pool = cfg.BACKBONE.EFFICIENTNET.FEATURE_AVG_POOL
        super(EfficientnetB6, self).__init__(outblocks, outplanes, outstrides, blocks_args, global_params,
                                             feature_jitter_scale,
                                             feature_jitter_prob, feature_avg_pool)
        if pretrained:
            self.load_pretrained_weights("efficientnet_b6", pretrained_model)
        self.shape_list = {"res{}".format(layer): {"channels": outplanes[layer - 1], "strides": outstrides[layer - 1]}
                           for layer in layers}

    def output_shape(self):
        return self.shape_list


class EfficientnetB5(EfficientNet):
    def __init__(self, cfg, pretrained, pretrained_model="", **override_params):
        layers = cfg.BACKBONE.EFFICIENTNET.LAYERS
        outblocks, outplanes, outstrides = get_outparams("efficientnet_b5", layers)
        blocks_args, global_params = get_model_params("efficientnet_b5", override_params)
        feature_jitter_scale = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_SCALE
        feature_jitter_prob = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_PROB

        feature_avg_pool = cfg.BACKBONE.EFFICIENTNET.FEATURE_AVG_POOL
        super(EfficientnetB5, self).__init__(outblocks, outplanes, outstrides, blocks_args=blocks_args,
                                             global_params=global_params, feature_jitter_scale=feature_jitter_scale,
                                             feature_jitter_prob=feature_jitter_prob,
                                             feature_avg_pool=feature_avg_pool)
        if pretrained:
            self.load_pretrained_weights("efficientnet_b5", pretrained_model)
        self.shape_list = {"res{}".format(layer): {"channels": outplanes[layer - 1], "strides": outstrides[layer - 1]}
                           for layer in layers}

    def output_shape(self):
        return self.shape_list


class EfficientnetB4(EfficientNet):
    def __init__(self, cfg, pretrained, pretrained_model="", **override_params):
        layers = cfg.BACKBONE.EFFICIENTNET.LAYERS
        outblocks, outplanes, outstrides = get_outparams("efficientnet_b4", layers)
        blocks_args, global_params = get_model_params("efficientnet_b4", override_params)
        feature_jitter_scale = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_SCALE
        feature_jitter_prob = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_PROB
        feature_avg_pool_size = cfg.BACKBONE.EFFICIENTNET.FEATURE_POOL_SIZE
        super(EfficientnetB4, self).__init__(outblocks, outplanes, outstrides, blocks_args, global_params,
                                             feature_jitter_scale,
                                             feature_jitter_prob, feature_avg_pool_size)
        if pretrained:
            self.load_pretrained_weights("efficientnet_b4", pretrained_model)
        self.shape_list = {"res{}".format(layer): {"channels": outplanes[layer - 1], "strides": outstrides[layer - 1]}
                           for layer in layers}

    def output_shape(self):
        return self.shape_list


class EfficientnetB3(EfficientNet):
    def __init__(self, cfg, pretrained, pretrained_model="", **override_params):
        layers = cfg.BACKBONE.EFFICIENTNET.LAYERS
        outblocks, outplanes, outstrides = get_outparams("efficientnet_b3", layers)
        blocks_args, global_params = get_model_params("efficientnet_b3", override_params)
        feature_jitter_scale = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_SCALE
        feature_jitter_prob = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_PROB
        feature_avg_pool = cfg.BACKBONE.EFFICIENTNET.FEATURE_AVG_POOL
        super(EfficientnetB3, self).__init__(outblocks, outplanes, outstrides, blocks_args, global_params,
                                             feature_jitter_scale, feature_jitter_prob,
                                             feature_avg_pool)
        if pretrained:
            self.load_pretrained_weights("efficientnet_b3", pretrained_model)
        self.shape_list = {"res{}".format(layer): {"channels": outplanes[layer - 1], "strides": outstrides[layer - 1]}
                           for layer in layers}

    def output_shape(self):
        return self.shape_list


class EfficientnetB2(EfficientNet):
    def __init__(self, cfg, pretrained, pretrained_model="", **override_params):
        layers = cfg.BACKBONE.EFFICIENTNET.LAYERS
        outblocks, outplanes, outstrides = get_outparams("efficientnet_b2", layers)
        blocks_args, global_params = get_model_params("efficientnet_b2", override_params)
        feature_jitter_scale = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_SCALE
        feature_jitter_prob = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_PROB
        feature_avg_pool = cfg.BACKBONE.EFFICIENTNET.FEATURE_AVG_POOL
        super(EfficientnetB2, self).__init__(outblocks, outplanes, outstrides, blocks_args, global_params,
                                             feature_jitter_scale, feature_jitter_prob,
                                             feature_avg_pool)
        if pretrained:
            self.load_pretrained_weights("efficientnet_b2", pretrained_model)
        self.shape_list = {"res{}".format(layer): {"channels": outplanes[layer - 1], "strides": outstrides[layer - 1]}
                           for layer in layers}

    def output_shape(self):
        return self.shape_list


class EfficientnetB1(EfficientNet):
    def __init__(self, cfg, pretrained, pretrained_model="", **override_params):
        layers = cfg.BACKBONE.EFFICIENTNET.LAYERS
        outblocks, outplanes, outstrides = get_outparams("efficientnet_b1", layers)
        blocks_args, global_params = get_model_params("efficientnet_b1", override_params)
        feature_jitter_scale = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_SCALE
        feature_jitter_prob = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_PROB
        feature_avg_pool = cfg.BACKBONE.EFFICIENTNET.FEATURE_AVG_POOL
        super(EfficientnetB1, self).__init__(outblocks, outplanes, outstrides, blocks_args, global_params,
                                             feature_jitter_scale, feature_jitter_prob,
                                             feature_avg_pool)
        if pretrained:
            self.load_pretrained_weights("efficientnet_b1", pretrained_model)
        self.shape_list = {"res{}".format(layer): {"channels": outplanes[layer - 1], "strides": outstrides[layer - 1]}
                           for layer in layers}

    def output_shape(self):
        return self.shape_list


class EfficientnetB0(EfficientNet):
    def __init__(self, cfg, pretrained, pretrained_model="", **override_params):
        layers = cfg.BACKBONE.EFFICIENTNET.LAYERS
        outblocks, outplanes, outstrides = get_outparams("efficientnet_b0", layers)
        blocks_args, global_params = get_model_params("efficientnet_b0", override_params)
        feature_jitter_scale = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_SCALE
        feature_jitter_prob = cfg.BACKBONE.EFFICIENTNET.FEATURE_JITTER_PROB
        feature_avg_pool = cfg.BACKBONE.EFFICIENTNET.FEATURE_AVG_POOL
        super(EfficientnetB0, self).__init__(outblocks, outplanes, outstrides, blocks_args, global_params,
                                             feature_jitter_scale, feature_jitter_prob,
                                             feature_avg_pool)
        if pretrained:
            self.load_pretrained_weights("efficientnet_b0", pretrained_model)
        self.shape_list = {"res{}".format(layer): {"channels": outplanes[layer - 1], "strides": outstrides[layer - 1]}
                           for layer in layers}

    def output_shape(self):
        return self.shape_list
