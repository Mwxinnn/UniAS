from math import log2, sqrt
import fvcore.nn.weight_init as weight_init
from torch import nn, Tensor
from ..modules.utils import Convx3Adaptor, MLP


class MGG_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__()
        self.conv0 = nn.Conv2d(
            input_dim, output_dim, kernel_size=1, stride=1, padding="same"
        )
        self.conv1 = nn.Conv2d(
            input_dim, output_dim, kernel_size=3, stride=1, padding="same"
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding="same"),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding="same"),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding="same"),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding="same"),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding="same"),
        )
        self.act = nn.GELU()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        feature_list = [
            self.act(self.conv0(x)),
            self.act(self.conv1(x)),
            self.act(self.conv2(x)),
            self.act(self.conv3(x)),
        ]
        out = feature_list[0] + feature_list[1] + feature_list[2] + feature_list[3]
        return out


class MGG_CNNLight(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__()
        assert hidden_dim == output_dim
        self.conv0 = nn.Conv2d(
            input_dim, output_dim, kernel_size=1, stride=1, padding="same"
        )
        self.conv1 = nn.Conv2d(
            input_dim, output_dim, kernel_size=3, stride=1, padding="same"
        )
        self.conv2 = nn.Conv2d(
            hidden_dim, output_dim, kernel_size=3, stride=1, padding="same"
        )
        self.conv3 = nn.Conv2d(
            hidden_dim, output_dim, kernel_size=3, stride=1, padding="same"
        )
        self.act = nn.GELU()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        feature_list = [
            self.act(self.conv0(x)),
            self.act(self.conv1(x)),
            self.act(self.conv2(self.conv1(x))),
            self.act(self.conv3(self.conv2(self.conv1(x)))),
        ]
        out = feature_list[0] + feature_list[1] + feature_list[2] + feature_list[3]
        return out


def get_adaptor(name, **cfg):
    if name == "MGG_CNN":
        return MGG_CNN(**cfg)
    elif name == "Convx3Adaptor":
        return Convx3Adaptor(**cfg)
    elif name == "Conv3Adaptor":
        return nn.Conv2d(
            in_channels=cfg["input_dim"],
            out_channels=cfg["output_dim"],
            kernel_size=3,
            stride=1,
            padding="same",
        )
    elif name == "Conv1Adaptor":
        return nn.Conv2d(
            in_channels=cfg["input_dim"],
            out_channels=cfg["output_dim"],
            kernel_size=1,
            stride=1,
            padding="same",
        )
    elif name == "MLP":
        return MLP(**cfg)
    elif name == "PlainAdaptor":
        return nn.Identity()
    else:
        raise NotImplementedError
