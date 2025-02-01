import torch.nn.functional as F
from torch import nn
from typing import Iterable

import fvcore.nn.weight_init as weight_init


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=16, norm_layer=None):
        super().__init__()
        self.patch_size = tuple(patch_size) if isinstance(patch_size, Iterable) else (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        weight_init.c2_xavier_fill(self.proj)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww) # TODO
        return x


class Convx3Adaptor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding="same"),
                                   nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding="same"),
                                   nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding="same"),
                                   # nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                   nn.GELU(),
                                   # nn.ReLU()
                                   )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        out = self.convs(x)
        return out


class Classifier(nn.Module):
    def __init__(self, feature_size, hidden_dim, num_class, **kwargs):
        super().__init__()
        h, w = feature_size
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(start_dim=1),
            nn.Linear(int(h * w / 4) * hidden_dim, num_class)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.layer(x)
