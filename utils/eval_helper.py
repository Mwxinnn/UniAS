import glob
import logging
import os
import cv2

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics


def dump(save_dir, outputs, input):
    filenames = input["filename"]
    batch_size = len(filenames)
    preds = outputs["pred"].cpu().numpy()  # B x 1 x H x W
    masks = input["mask"].cpu().numpy()  # B x 1 x H x W
    heights = input["height"].cpu().numpy()
    widths = input["width"].cpu().numpy()
    clsnames = input["clsname"]
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file,
            filename=filenames[i],
            pred=preds[i],
            mask=masks[i],
            height=heights[i],
            width=widths[i],
            clsname=clsnames[i],
        )


def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    preds = []
    masks = []
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "height": npz["height"],
                "width": npz["width"],
                "clsname": str(npz["clsname"]),
            }
        )
        preds.append(npz["pred"])
        masks.append(npz["mask"])
    preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    return fileinfos, preds, masks


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(int)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds, top_percent):
        N, _, _ = preds.shape
        pred = preds.reshape(N, -1)
        if top_percent == 1:
            return pred.std(axis=1)  # (N, )
        else:
            index = int(pred.shape[1] * top_percent)
            return np.sort(pred, axis=1)[:, index:].std(axis=1)


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size, top_percent):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        )  # N x 1 x H x W
        pred = preds.reshape(N, -1)
        if top_percent == 1:
            return pred.max(axis=1)  # (N, )
        else:
            index = int(pred.shape[1] * top_percent)
            return np.sort(pred, axis=1)[:, index:].sum(axis=1)


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        self.auc_threshod = thresholds[np.argmax(tpr - fpr)]
        return auc

    def eval_dice_and_iou(self, data_meta,save_dir,files):
        precision, recall, thresholds = metrics.precision_recall_curve(self.masks, self.preds)
        
        best_th = thresholds[np.argmax(precision + recall)]

        dice_list = []
        iou_list = []

        for img_idx in range(data_meta.preds.shape[0]):
            img_pred = (data_meta.preds[img_idx] >= best_th).astype(int)
            mask = (data_meta.masks[img_idx]!=0).astype(int)
            if mask.sum() == 0 and img_pred.sum() == 0:
                dice = iou = 1
            if mask.sum() == 0 and img_pred.sum() != 0:
                dice = iou = 0
            if mask.sum() != 0:
                intersection = np.sum(mask * img_pred)
                total = np.sum(mask) + np.sum(img_pred)
                dice = 2 * intersection / total
                iou = intersection / (total - intersection)
            dice_list.append(dice)
            iou_list.append(iou)
        return dice_list, iou_list

    def eval_ap(self):
        ap = metrics.average_precision_score(self.masks, self.preds, pos_label=1)
        return ap


eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
}


def performances(fileinfos, preds, masks, config, save_dir):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        files = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                files.append(fileinfo["filename"])
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
        # preds_cls=(preds_cls-preds_cls.min())/(preds_cls.max()-preds_cls.min())
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
        data_meta = EvalDataMeta(preds_cls, masks_cls)

        # auc
        if config.get("AUC", None):
            for metric in config.AUC:
                evalname = metric["NAME"]
                if evalname in ["ap", "dice"]:
                    continue
                kwargs = metric.get("KWARGS", {})
                eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
                auc = eval_method.eval_auc()
                ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc
                if evalname == "pixel":
                    if {"NAME": "dice"} in config.AUC:
                        dice, iou = eval_method.eval_dice_and_iou(data_meta,save_dir,files)
                        ret_metrics["{}_dice_auc".format(clsname)] = np.mean(dice)
                        ret_metrics["{}_iou_auc".format(clsname)] = np.mean(iou)
                    if {"NAME": "ap"} in config.AUC:
                        ret_metrics["{}_ap_auc".format(clsname)] = eval_method.eval_ap()

    if config.get("AUC", None):
        for metric in config.AUC:
            evalname = metric["NAME"]
            evalvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            dicevalues = [ret_metrics["{}_dice_auc".format(clsname)] for clsname in clsnames]
            iouvalues = [ret_metrics["{}_iou_auc".format(clsname)] for clsname in clsnames]
            apvalues = [ret_metrics["{}_ap_auc".format(clsname)] for clsname in clsnames]
            mean_auc = np.mean(np.array(evalvalues))
            mean_dice = np.mean(np.array(dicevalues))
            mean_iou = np.mean(np.array(iouvalues))
            mean_ap = np.mean(np.array(apvalues))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc
            ret_metrics["{}_dice_auc".format("mean")] = mean_dice
            ret_metrics["{}_iou_auc".format("mean")] = mean_iou
            ret_metrics["{}_ap_auc".format("mean")] = mean_ap
    return ret_metrics


def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]
    print(len(clsnames))

    # auc
    if config.get("AUC", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        logger.info(f"\n{record}")
