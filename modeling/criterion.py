import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["NAME"]
        loss_dict[loss_name] = globals()[cfg["TYPE"]](**cfg["KWARGS"])
    return loss_dict


class MSE_loss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""

    def __init__(self, weight=0.5, feature_levels=4):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight
        self.feature_levels = feature_levels

    def forward(self, input, label):
        """

        :param input: 网络输出的字典
        :param label: 网络backbone输出特征的第一个特征 tensor
        :return:
        """
        # TODO
        features = input["recon_feature"]
        labels = [label["res{}".format(self.feature_levels - i % self.feature_levels)] for i in
                  range(1, len(features))]

        deepest_f = features[0]
        deepest_l = label["res4"]
        loss = self.criterion_mse(deepest_f, deepest_l) * 10

        for cur_feature, cur_label in zip(features[1:], labels):
            loss += self.criterion_mse(cur_feature, cur_label)
        loss *= self.weight
        # if not label["contaminated"]:
        #     loss *= 2
        return loss


class CosSim_loss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""

    def __init__(self, weight=0.5, feature_levels=4):
        super().__init__()
        self.criterion_cos = nn.CosineSimilarity()
        self.weight = weight
        self.feature_levels = feature_levels

    def forward(self, input, label):
        """

        :param input: 网络输出的字典
        :param label: 网络backbone输出特征的第一个特征 tensor
        :return:
        """
        # TODO
        features = input["recon_feature"]
        labels = [label["res{}".format(self.feature_levels - i % self.feature_levels)] for i in
                  range(1, len(features))]

        deepest_f = features[0]
        deepest_l = label["res4"]
        loss = (1 - torch.mean(self.criterion_cos(deepest_f.reshape(deepest_f.shape[0], -1),
                                                  deepest_l.reshape(deepest_l.shape[0], -1)))) * 10

        for cur_feature, cur_label in zip(features[1:], labels):
            loss += 1 - torch.mean(self.criterion_cos(cur_feature.reshape(cur_feature.shape[0], -1),
                                                      cur_label.reshape(cur_label.shape[0], -1)))
        loss *= self.weight
        # if not label["contaminated"]:
        #     loss *= 2
        return loss