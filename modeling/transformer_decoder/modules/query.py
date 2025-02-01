import torch
from torch import nn

from ..modules.adaptor import MGG_CNN
from ..modules.utils import MLP, Convx3Adaptor


class ChannelAttentionCNN(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.agg = MLP(input_dim, input_dim // 2, output_dim, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        avg = self.agg(self.avg_pooling(feature).squeeze(2, 3))
        max = self.agg(self.max_pooling(feature).squeeze(2, 3))
        return self.sigmoid(avg + max)


class SpatialAttentionCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        avg = torch.mean(feature, 1, keepdim=True)
        max, _ = torch.max(feature, 1, keepdim=True)
        feature_map = torch.cat([avg, max], dim=1)
        out = self.sigmoid(self.conv(feature_map))
        return out


class LinCombWeightedQuery(nn.Module):
    def __init__(
        self,
        init_num_queries,
        num_queries,
        hidden_dim,
        feature_size=(14, 14),
        **kwargs,
    ):
        super().__init__()
        self.channel_weight = ChannelAttentionCNN(hidden_dim, hidden_dim)
        self.spatial_weight = SpatialAttentionCNN()

        self.num_queries = num_queries
        self.ratio = init_num_queries // num_queries

        self.adaptor = MGG_CNN(hidden_dim * self.ratio, hidden_dim, hidden_dim)
        self._reset_parameters()

        self.feature_size = feature_size

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature, s_query, **kwargs):
        # s_query [196*n,256]
        spatial_enhanced_feature = torch.einsum(
            "bh,lbh->lbh", self.channel_weight(feature), s_query
        )
        spatial_weight = self.spatial_weight(feature).flatten(2).squeeze(1)
        channel_enhanced_feature = torch.einsum(
            "bl,lbh->lbh",
            spatial_weight.repeat(1, self.ratio),
            s_query,
        )
        output = channel_enhanced_feature + spatial_enhanced_feature
        output = output.permute((1, 2, 0))
        output = output.reshape(
            output.shape[0],
            output.shape[1] * self.ratio,
            self.feature_size[0],
            self.feature_size[1],
        )
        output = self.adaptor(output).flatten(2).permute((2, 0, 1))
        return output


def get_query_processor(name, **cfg):
    if name == "LinCombWeightedQuery":
        return LinCombWeightedQuery(**cfg)
    else:
        raise NotImplementedError


def get_query_projector(name, in_dim, out_dim, **cfg):
    if name == "1Conv1x1":
        return nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding="same")
    elif name == "1Conv3x3":
        return nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding="same")
    elif name == "3Conv3x3":
        assert "hidden_dim" in cfg.keys(), "hidden_dim not specified!"
        return Convx3Adaptor(in_dim, cfg["hidden_dim"], out_dim)
    elif name == "MLP3":
        assert "hidden_dim" in cfg.keys(), "hidden_dim not specified!"
        return MLP(in_dim, cfg["hidden_dim"], out_dim, 3)
    elif name == "MLP1":
        return MLP(in_dim, out_dim, out_dim, 1)
    elif name == "Identity" or name == "linCombWeight":
        return nn.Identity()
    else:
        raise NotImplementedError
