import logging

from datasets.dataset_loader import build_custom_dataloader

logger = logging.getLogger("global")


def build(cfg, training, distributed):
    if training:
        cfg.update(cfg.get("TRAIN", {}))
    else:
        cfg.update(cfg.get("TEST", {}))

    dataset = cfg["TYPE"]
    if dataset == "anomaly":
        data_loader = build_custom_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, distributed=True, train=True):
    train_loader = None
    if train:
        if cfg_dataset.get("TRAIN", None):
            train_loader = build(cfg_dataset, training=True, distributed=distributed)

    test_loader = None
    if cfg_dataset.get("TEST", None):
        test_loader = build(cfg_dataset, training=False, distributed=distributed)

    logger.info("build datasets done")
    return train_loader, test_loader
