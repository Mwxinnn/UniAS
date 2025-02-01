"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).
import os
import random
import pdb
import torch
from detectron2.modeling import ShapeSpec
from kornia.filters import gaussian_blur2d
from torch import nn
from torch.nn import functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import logging

logger = logging.getLogger("global_logger")

from .utils import (
    MemoryEfficientSwish,
    Swish,
    calculate_output_image_size,
    drop_connect,
    efficientnet_params,
    get_same_padding_conv2d,
    round_filters,
    round_repeats,
    MODEL_URLS_ADVPROP,
    MODEL_URLS,
)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = (
                1 - global_params.batch_norm_momentum
        )  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
                0 < self._block_args.se_ratio <= 1
        )
        self.id_skip = (
            block_args.id_skip
        )  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = (
                self._block_args.input_filters * self._block_args.expand_ratio
        )  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            )
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
        )
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        )
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        if (
                self.id_skip
                and self._block_args.stride == 1
                and input_filters == output_filters
        ):
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    # Example:
    #     >>> import torch
    #     >>> from efficientnet.model import EfficientNet
    #     >>> inputs = torch.rand(1, 3, 224, 224)
    #     >>> model = EfficientNet.from_pretrained('efficientnet-b0')
    #     >>> model.eval()
    #     >>> outputs = model(inputs)
    """

    def __init__(self, outblocks, outplanes, outstrides, blocks_args=None, global_params=None, feature_jitter_scale=20,
                 feature_jitter_prob=1, feature_avg_pool_size=0):
        """

        """
        super(EfficientNet, self).__init__()
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.feature_jitter_scale = feature_jitter_scale
        self.feature_jitter_prob = feature_jitter_prob
        self.avg_pool = (feature_avg_pool_size != 0)
        self.kernel_size = feature_avg_pool_size
        self.sigma = 1

        self.outblocks = {"res{}".format(i + 1): outblocks[i] for i in range(len(outblocks))}
        self._out_feature_channels = {"res{}".format(i + 1): outplanes[i] for i in range(len(outplanes))}
        self._out_feature_strides = {"res{}".format(i + 1): outstrides[i] for i in range(len(outstrides))}

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(
            32, self._global_params
        )  # number of output channels
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params
                ),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(
                MBConvBlock(block_args, self._global_params, image_size=image_size)
            )
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(block_args, self._global_params, image_size=image_size)
                )
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

        ##################################################################################################
        # construct self.outplanes
        inputs = torch.ones((1, 3, 256, 256))
        _, _, outplane_dict = self.extract_features(inputs)
        self.num_features = [outplane_dict[i] for i in outblocks]

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        ###########################################################
        feat_dict = {}
        outplane_dict = {}
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            ##############################################################
            feat_dict[idx] = x
            outplane_dict[idx] = x.shape[1]  # x: b x c x h x w
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x, feat_dict, outplane_dict

    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            bs, channels, height, width = feature_tokens.shape
            feature_norms = (
                    feature_tokens.norm(dim=1).unsqueeze(1) / channels
            )  # (H x W) x B x 1
            jitter = torch.randn((bs, channels, height, width)).cuda()
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, inputs, cls_idx=None, training=True):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        image = inputs
        # Convolution layers
        x, feat_dict, _ = self.extract_features(image)
        # get features for training & test
        self.features = {"res{}".format(i + 1): feat_dict[self.outblocks["res{}".format(i + 1)]] for i in
                         range(len(self.outblocks.values()))}
        # pdb.set_trace()
        if not self.avg_pool:
            self.noise_features = {
                key: self.add_jitter(self.features[key], self.feature_jitter_scale, self.feature_jitter_prob, )
                for key in self.features.keys()}
            self.features["clsidx"] = cls_idx
            return self.features, self.noise_features
        # gaussian filtering
        assert self.avg_pool, "only pooled features are allowed!"
        self.forward_features = {
            k: torch.cat([gaussian_blur2d(v, (self.kernel_size, self.kernel_size), (self.sigma, self.sigma)),
                        v - gaussian_blur2d(v, (self.kernel_size, self.kernel_size), (self.sigma, self.sigma))],
                        dim=1)
            for k, v in self.features.items()}
        
        # self.forward_features = {
        #     k: nn.functional.avg_pool2d(v, kernel_size=self.kernel_size, stride=1,padding=1) for k, v in
        #     self.features.items()}
        # self.features = {k: nn.functional.avg_pool2d(v, kernel_size=self.kernel_size, stride=1,padding=1) for k, v
        #     in self.features.items()}
        
        # self.forward_features = {
        #     k: gaussian_blur2d(v, (self.kernel_size, self.kernel_size), (self.sigma, self.sigma)) for k, v in
        #     self.features.items()}
        self.features = {k: gaussian_blur2d(v, (self.kernel_size, self.kernel_size), (self.sigma, self.sigma)) for k, v
                        in self.features.items()}
        self.features["clsidx"] = cls_idx
        # returen with no noise
        if not training:
            return self.features, self.forward_features
        # return with noise
        self.noise_features = {
            key: self.add_jitter(self.forward_features[key], self.feature_jitter_scale, self.feature_jitter_prob, )
            for key in self.forward_features.keys()}
        # return both clean and noisy features
        return self.features, self.noise_features

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.features
        }

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, bias=False
            )

    def load_pretrained_weights(self, model_name, pretrained_model="", load_fc=False, advprop=False):
        """Loads pretrained weights from weights path or download using url.

        Args:
            model (Module): The whole model of efficientnet.
            model_name (str): Model name of efficientnet.
            pretrained_model (str):
                str: path to pretrained weights file on the local disk.
                if not exist: pretrained weights downloaded from the Internet.
            load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
            advprop (bool): Whether to load pretrained weights
                            trained with advprop (valid when pretrained_model is None).
        """
        if os.path.exists(pretrained_model):
            state_dict = torch.load(pretrained_model)
        else:
            urls = MODEL_URLS_ADVPROP if advprop else MODEL_URLS
            logger.info(
                "{} not exist, load from {}".format(pretrained_model, urls[model_name])
            )
            state_dict = load_state_dict_from_url(urls[model_name], progress=True)

        if load_fc:
            ret = self.load_state_dict(state_dict=state_dict, strict=False)
            assert (
                not ret.missing_keys
            ), "Missing keys when loading pretrained weights: {}".format(ret.missing_keys)
        else:
            state_dict.pop("_fc.weight")
            state_dict.pop("_fc.bias")
            ret = self.load_state_dict(state_dict=state_dict, strict=False)
            assert set(ret.missing_keys) == set(
                ["_fc.weight", "_fc.bias"]
            ), "Missing keys when loading pretrained weights: {}".format(ret.missing_keys)
        assert (
            not ret.unexpected_keys
        ), "Missing keys when loading pretrained weights: {}".format(ret.unexpected_keys)

        logger.info("Loaded ImageNet pretrained {}".format(model_name))
