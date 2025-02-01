from __future__ import division

import json
import logging

import numpy as np
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

logger = logging.getLogger("global_logger")

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import datasets.transforms as T


class BaseDataset(Dataset):
    """
    A datasets should implement
        1. __len__ to get size of the datasets, Required
        2. __getitem__ to get a single data, Required

    """

    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TrainBaseTransform(object):
    """
    Resize, flip, rotation for image and mask
    """

    def __init__(self, input_size, hflip, vflip, rotate):
        self.input_size = input_size  # h x w
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate

    def __call__(self, image, mask, cls_name):
        # transform_fn = transforms.Resize(self.input_size, Image.BILINEAR)
        # image = transform_fn(image)
        # transform_fn = transforms.Resize(self.input_size, Image.NEAREST)
        # mask = transform_fn(mask)
        if self.hflip and cls_name in self.hflip:
            transform_fn = T.RandomHFlip()
            image, mask = transform_fn(image, mask)
        if self.vflip and cls_name in self.vflip:
            transform_fn = T.RandomVFlip()
            image, mask = transform_fn(image, mask)
        if self.rotate and cls_name in self.rotate:
            transform_fn = T.RandomRotation([0, 90, 180, 270])
            image, mask = transform_fn(image, mask)
        return image, mask


class TestBaseTransform(object):
    """
    Resize for image and mask
    """

    def __init__(self, input_size):
        self.input_size = input_size  # h x w

    def __call__(self, image, mask, **kwargs):
        # transform_fn = transforms.Resize(self.input_size, Image.BILINEAR)
        # image = transform_fn(image)
        # transform_fn = transforms.Resize(self.input_size, Image.NEAREST)
        # mask = transform_fn(mask)
        return image, mask


def build_custom_dataloader(cfg, training, distributed=True):
    image_reader = build_image_reader(cfg.IMAGE_READER)

    normalize_fn = transforms.Normalize(mean=cfg["PIXEL_MEAN"], std=cfg["PIXEL_STD"])
    if training:
        transform_fn = TrainBaseTransform(
            cfg["INPUT_SIZE"], cfg["HFLIP"], cfg["VFLIP"], cfg["ROTATE"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["INPUT_SIZE"])

    colorjitter_fn = None
    if cfg.get("COLORJITTER", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["COLORJITTER"])

    logger.info("building CustomDataset from: {}".format(cfg["META_FILE"]))

    dataset = MVTecDataset(
        image_reader,
        cfg["META_FILE"],
        cfg["CLS_IDX"],
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["BATCH_SIZE"],
        num_workers=cfg["WORKERS"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader


class MVTecDataset(BaseDataset):
    def __init__(
            self,
            image_reader,
            meta_file,
            cls_idx,
            transform_fn,
            normalize_fn,
            colorjitter_fn=None,
    ):
        super(MVTecDataset, self).__init__()
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn

        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

        with open(cls_idx, "r") as f_r:
            self.cls_idx = json.load(f_r)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = meta["filename"]
        label = meta["label"]
        image = self.image_reader(meta["filename"])
        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]
        input["clsidx"] = self.cls_idx[input["clsname"]]

        image = Image.fromarray(image, "RGB")

        # read / generate mask
        if meta.get("maskname", None):
            mask = self.image_reader(meta["maskname"], is_mask=True)
        else:
            if label == 0:  # good
                mask = np.zeros((image.height, image.width)).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        mask = Image.fromarray(mask, "L")

        if self.transform_fn:
            image, mask = self.transform_fn(image, mask, cls_name=input["clsname"])
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        if self.normalize_fn:
            image = self.normalize_fn(image)
        input.update({"image": image, "mask": mask})
        return input
