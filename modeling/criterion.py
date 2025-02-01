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


class output_classification_ce(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, input, label):
        preds = input["cls_preds"]
        labels = label["clsidx"]
        loss = torch.tensor(0.).cuda()
        for pred in preds:
            loss += self.ce(pred, labels)
        loss *= self.weight
        return loss


# TODO
class classification_ce(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, reconstructed, original):
        """

        :param reconstructed: 输入单张图像重建结果
        :param original: 原图
        :return: MSEloss
        """
        cls_pred = reconstructed["feature_pred_class"]
        cls_label = original["clsidx"]
        loss = self.ce(cls_pred, cls_label) * self.weight
        # print("classification CE:", loss.item())
        return loss


class classification_query(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, reconstructed, original):
        """

        :param reconstructed: 输入单张图像重建结果
        :param original: 原图
        :return: MSEloss
        """
        cls_pred = reconstructed["query_pred_class"]
        cls_label = original["clsidx"]
        loss = self.ce(cls_pred, cls_label)
        return loss * self.weight


class class_token_norm_ce(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, reconstructed, original):
        """

        :param reconstructed: 输入单张图像重建结果
        :param original: 原图
        :return: MSEloss
        """
        cls_pred = reconstructed["class_token"]
        loss = 0.
        for i in range(cls_pred.shape[0]):
            loss += self.ce(cls_pred[i], torch.tensor(i).cuda())
        loss = loss * self.weight
        # print("classification CE class token:", loss.item())
        return loss


class entropy_normalized(nn.Module):
    def __init__(self, weight_entropy=1, weight_div=1):
        super().__init__()
        self.weight_entropy = weight_entropy
        self.weight_div = weight_div
        self.div = nn.KLDivLoss()

    def forward(self, reconstructed, original):
        query_s, query_c = reconstructed["queries"]
        s_entropy = - torch.mean(query_s * torch.log2(query_s))
        c_entropy = - torch.mean(query_c * torch.log2(query_c))
        div_loss = self.div(query_s, query_c)
        return (s_entropy + c_entropy) * self.weight_entropy + div_loss * self.weight_div


class alignment_loss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        # self.bce = nn.BCELoss(reduction="mean")
        # self.bce = nn.CrossEntropyLoss(reduction="mean")
        self.weight = weight

    def forward(self, reconstructed, original):
        """

        :param reconstructed: 输入单张图像重建结果
        :param original: 原图
        :return: MSEloss
        """
        pred = reconstructed["pred"]
        entropy = pred.mean()
        return entropy * self.weight


class denoise_loss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def forward(self, reconstructed, label):
        preds = reconstructed['denoise_masks']
        loss = 0.
        for pred in preds:
            loss += pred.mean()
        return loss * self.weight


class dice_loss(nn.Module):
    def __init__(self, weight=1, smooth=1):
        super().__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self, pred, label):
        pred = pred["pred"]
        label = label["mask"]
        label = label.to(torch.device("cuda"))
        label = nn.functional.interpolate(label, pred.shape[-2:], mode="nearest")
        num = label.size(0)

        probs = F.sigmoid(pred)
        m1 = probs.view(num, -1)
        m2 = label.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        print("dice", score.item())
        return score


class focal_loss(nn.Module):
    def __init__(self, weight=1, alpha=-1, gamma=4):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, label):
        pred = pred["pred"].sigmoid()
        label = label["mask"]
        label = label.to(torch.device("cuda"))
        label = nn.functional.interpolate(label, pred.shape[-2:])
        ce_loss = self.bce(pred, label)
        p_t = pred * label + (1 - pred) * (1 - label)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * label + (1 - self.alpha) * (1 - label)
            loss = alpha_t * loss
        loss = loss.mean() * self.weight
        return loss
