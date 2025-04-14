# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
from math import sqrt, log2

import fvcore.nn.weight_init as weight_init
import torch
from kornia.filters import gaussian_blur2d
from torch import nn

from ..transformer_decoder.modules.adaptor import get_adaptor
from ..transformer_decoder.modules.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.modules.query import get_query_processor, get_query_projector
from ..transformer_decoder.modules.transformer import (
    get_transformer_block,
    get_ablation_transformer_block,
)
from ..transformer_decoder.modules.utils import PatchEmbed, Classifier


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
        self, model_cfg, input_shape: dict, feature_size: list, **kwargs
    ) -> None:
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        self.num_feature_levels = len(model_cfg.SEM_SEG_HEAD.IN_FEATURES)
        self.loss_cfg = model_cfg.get("CRITERION", False)
        assert self.loss_cfg, "loss criterion must be specified!"

        input_shape = sorted(
            input_shape.items(), key=lambda x: x[1]["strides"]
        )  # 分辨率由高到低
        self.strides = [v["strides"] for k, v in input_shape]
        self.channels = [v["channels"] for k, v in input_shape]

        # ===============================
        # patch embedding
        in_dim = model_cfg.SEM_SEG_HEAD.CONV_DIM
        self.hidden_dim = model_cfg.MASK_FORMER.HIDDEN_DIM
        self.patch_embed = nn.ModuleList()
        for layer in range(self.num_feature_levels):
            patch_size = self.strides[layer] // 2
            if patch_size == 1:
                self.patch_embed.append(nn.Identity())
            else:
                self.patch_embed.append(
                    PatchEmbed(
                        patch_size,
                        embed_dim=self.hidden_dim * (layer + 1) // 2,
                        in_chans=in_dim,
                    )
                )

        # ===============================
        # positional encoding
        N_steps = self.hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        # ===============================
        # define Transformer decoder here
        transformer_block_cfg = model_cfg.MASK_FORMER["ATTN"]
        self.num_queries = model_cfg.MASK_FORMER.NUM_OBJECT_QUERIES
        transformer_block_cfg["spatial_hdim"] = self.hidden_dim
        if "Channel" in transformer_block_cfg.NAME:
            transformer_block_cfg["channel_hdim"] = self.num_queries
        self.transformer_block = nn.ModuleList(
            [
                get_transformer_block(
                    transformer_block_cfg.NAME, **transformer_block_cfg
                )
                for _ in range(self.num_feature_levels)
            ]
        )
        # prepare masks for transformer
        self.neighbor_mask = model_cfg.MASK_FORMER.get("NEIGHBOR_MASK", None)
        self.self_attn_mask, self.cross_attn_mask = self.get_masks_for_transformer(
            feature_size, self.neighbor_mask
        )

        # read config
        query_cfg = model_cfg.MASK_FORMER.get("QUERY", None)
        feature_projector_cfg = query_cfg.FEATURE_PROJECTOR
        processor_cfg = query_cfg.PROCESSOR
        adaptor_cfg = model_cfg.MASK_FORMER.get("ADAPTOR", None)
        if processor_cfg.NAME == "LinCombWeightedQuery":
            init_num_queries = processor_cfg.base_ratio * self.num_queries
            processor_cfg["init_num_queries"] = init_num_queries
        else:
            init_num_queries = self.num_queries
        # ===============================
        # learnable query features
        self.spatial_query_feat = nn.Embedding(init_num_queries, self.hidden_dim)
        # learnable query p.e.
        self.spatial_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

        # process learnable querys
        processor_cfg["num_queries"] = self.num_queries
        self.feature_projector = get_query_projector(
            feature_projector_cfg.NAME, in_dim, self.hidden_dim, **feature_projector_cfg
        )
        self.query_processor = get_query_processor(processor_cfg.NAME, **processor_cfg)
        # ===============================
        # prepare src
        # level embedding (we always use 3 scales)
        self.level_embed_src = nn.Embedding(self.num_feature_levels, self.hidden_dim)
        self.input_proj = nn.ModuleList()
        for layer in range(self.num_feature_levels):
            if layer != 0:
                self.input_proj.append(
                    nn.Conv2d(
                        self.hidden_dim * (layer + 1) // 2,
                        self.hidden_dim,
                        kernel_size=1,
                        stride=1,
                        padding="same",
                    )
                )
            else:
                self.input_proj.append(
                    nn.Conv2d(
                        self.hidden_dim,
                        self.hidden_dim,
                        kernel_size=1,
                        stride=1,
                        padding="same",
                    )
                )
            weight_init.c2_xavier_fill(self.input_proj[-1])

        # ===============================
        # define layer adaptors
        adaptor_cfg["input_dim"] = self.hidden_dim
        adaptor_cfg["output_dim"] = self.hidden_dim
        self.layer_adaptors = nn.ModuleList(
            [
                get_adaptor(name=adaptor_cfg.NAME, **adaptor_cfg)
                for _ in range(self.num_feature_levels)
            ]
        )
        # ===============================
        # output FFNs
        mask_dim = model_cfg.MASK_FORMER.MASK_DIM
        self.decoder_norm = nn.LayerNorm(self.hidden_dim)
        self.out_agg = nn.ModuleList()
        for stride, channel in zip(self.strides, self.channels[::-1]):
            h, w = feature_size
            original_size = (int(h * stride // 2), int(w * stride // 2))
            self.out_agg.append(  # TODO
                nn.Sequential(
                    nn.UpsamplingBilinear2d(
                        size=original_size
                    ),  # 插值到1 2 4 8倍，分辨率分别为14，28，56，112
                    nn.Conv2d(
                        self.hidden_dim,
                        mask_dim,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                    ),
                    nn.Conv2d(
                        mask_dim, mask_dim, kernel_size=3, stride=1, padding="same"
                    ),
                    nn.Conv2d(
                        mask_dim, channel, kernel_size=3, stride=1, padding="same"
                    ),
                )
            )
        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=self.strides[0])
        self.post_gaussian = model_cfg.MASK_FORMER.get("POST_GAUSSIAN", False)
        if self.post_gaussian:
            self.gaussian_parm_list = []
            for size in range(3, 10, 2):
                # self.gaussian_parm_list.append((3, 1))
                self.gaussian_parm_list.append((size, size / 3))

    def forward(self, x, feature_dict):
        # x分辨率由低到高
        # x is a list of multi-scale feature, mask_feature is the reconstructed version of x
        assert len(x) == self.num_feature_levels
        assert sqrt(self.num_queries) == int(sqrt(self.num_queries))
        self.loss_dict = {"RECON_LOSS": torch.tensor(0.0).cuda()}

        recon_size = int(sqrt(self.num_queries))
        feature_list = [
            feature_dict["res{}".format(i)]
            for i in range(self.num_feature_levels, 0, -1)
        ]

        src = []
        pos = []
        bs, c, h, w = x[0].shape
        for i in range(self.num_feature_levels):
            # process input feature and positional embedding
            projected_feature = self.input_proj[i](self.patch_embed[i](x[i]))
            pos.append(self.pe_layer(projected_feature, None).flatten(2))
            src.append(
                projected_feature.flatten(2)
                + self.level_embed_src.weight[i][None, :, None]
            )
            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        # prepare feature
        # QxNxC
        spatial_query_embed = self.spatial_query_embed.weight.unsqueeze(1).repeat(
            1, bs, 1
        )
        spatial_output = self.spatial_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        # BxCxHxW->BxCxHW -> BxHWxC -> HWxBxC
        feature = self.feature_projector(x[0])
        output = self.query_processor(
            feature, s_query=spatial_output
        )

        recon_feature = []
        anomaly_maps = []
        # prediction heads on learnable query features
        # start transformer
        for i in range(self.num_feature_levels):
            level_index = i % self.num_feature_levels
            # ========================================================
            output = self.transformer_block[level_index](
                output,
                src[level_index],
                crossattn_mask=self.cross_attn_mask,
                selfattn_mask=self.self_attn_mask,
                pos=pos[level_index],
                spatial_query_pos=spatial_query_embed,
            )

            output = (
                self.layer_adaptors[level_index](
                    output.permute((1, 2, 0)).reshape(bs, -1, recon_size, recon_size)
                )
                .flatten(2)
                .permute((2, 0, 1))
            )

            outputs_mask, anomaly_map = self.forward_prediction_heads(
                output,
                (recon_size, recon_size),
                level_index,
                feature_list,
            )
            recon_feature.append(outputs_mask)
            anomaly_maps.append(anomaly_map)

        seg_output = (anomaly_maps[-1], anomaly_maps[:-1])
        assert (
            len(recon_feature) == self.num_feature_levels
        )  # 每一层后面取一个输出加一个监督
        out = {
            "recon_feature": recon_feature,
            "pred": seg_output,
        }
        out = self.get_final_pred(out)
        return out

    def forward_prediction_heads(self, output, size, index, feature_list):
        """
        对每一层特征的输出都重建一次最底层的特征图
        """
        recon_feature = output

        decoder_output = self.decoder_norm(recon_feature)
        c, bs, _ = decoder_output.shape
        decoder_output = (
            decoder_output.transpose(0, 1)
            .permute((0, 2, 1))
            .reshape((bs, -1, size[0], size[1]))
        )
        outputs_mask = self.out_agg[index](decoder_output)
        # anomaly map
        anomaly_map_target_size = feature_list[-1].shape[-2:]
        anomaly_map, recon_loss = self.get_anomaly_map_and_recon_loss(
            outputs_mask, feature_list[index], anomaly_map_target_size
        )
        self.loss_dict["RECON_LOSS"] += recon_loss
        if self.post_gaussian:
            kernel, sigma = self.gaussian_parm_list[index]
            anomaly_map = gaussian_blur2d(
                anomaly_map.unsqueeze(1),
                kernel_size=(kernel, kernel),
                sigma=(sigma, sigma),
            ).squeeze(1)

        return outputs_mask, anomaly_map  # before activated

    def get_anomaly_map_and_recon_loss(self, out, feature, target_size):
        anomaly_map = 1 - nn.functional.cosine_similarity(out, feature)

        # recon loss
        recon_loss = torch.tensor(0.0).cuda()
        if "MSE_loss" in self.loss_cfg.RECON_LOSS.get("NAME", None):
            recon_loss = recon_loss + nn.functional.mse_loss(out, feature)
        if "CosSim_loss" in self.loss_cfg.RECON_LOSS.get("NAME", None):
            recon_loss = recon_loss + anomaly_map.mean()

        if anomaly_map.shape[-2:] != target_size:
            anomaly_map = nn.functional.interpolate(
                anomaly_map.unsqueeze(1), target_size, mode="bilinear"
            ).squeeze(1)
        return anomaly_map, recon_loss

    def generate_mask(self, feature_size, neighbor_size):
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

    def get_masks_for_transformer(self, feature_size, model_cfg):
        if model_cfg is not None:
            # TODO
            self_attn_mask = self.generate_mask(
                feature_size, self.neighbor_mask.NEIGHBOR_SIZE
            )
            cross_attn_mask = self.generate_mask(
                feature_size, self.neighbor_mask.NEIGHBOR_SIZE
            )
        else:
            self_attn_mask = cross_attn_mask = None
        return self_attn_mask, cross_attn_mask

    def get_final_pred(self, outputs):
        anomaly_map, aux_anomaly_maps = outputs["pred"]
        for aux_anomaly_map in aux_anomaly_maps:
            anomaly_map *= aux_anomaly_map
        pred = self.upsampler(anomaly_map.unsqueeze(1))
        outputs["pred"] = pred

        loss = 0.0
        for key, value in self.loss_dict.items():
            loss += self.loss_dict[key] * self.loss_cfg[key].weight
        outputs["loss"] = loss
        return outputs
