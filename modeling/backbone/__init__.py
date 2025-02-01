from .efficientnet import *  # noqa F401

# from .resnet import *  # noqa F401



def build_backbone(cfg):
    name = cfg.BACKBONE.NAME
    pretrained = cfg.BACKBONE.get("PRETRAINED", False)
    pretrained_model = cfg.BACKBONE.get("PRETRAINED_WEIGHT_PATH", None)
    assert name == "EfficientnetB4", "Only EfficientnetB4 is supported"
    return EfficientnetB4(cfg, pretrained, pretrained_model)


backbone_info = {
    "resnet18": {
        "layers": [1, 2, 3, 4],
        "planes": [64, 128, 256, 512],
        "strides": [4, 8, 16, 32],
    },
    "resnet34": {
        "layers": [1, 2, 3, 4],
        "planes": [64, 128, 256, 512],
        "strides": [4, 8, 16, 32],
    },
    "resnet50": {
        "layers": [1, 2, 3, 4],
        "planes": [256, 512, 1024, 2048],
        "strides": [4, 8, 16, 32],
    },
    "resnet101": {
        "layers": [1, 2, 3, 4],
        "planes": [256, 512, 1024, 2048],
        "strides": [4, 8, 16, 32],
    },
    "wide_resnet50_2": {
        "layers": [1, 2, 3, 4],
        "planes": [256, 512, 1024, 2048],
        "strides": [4, 8, 16, 32],
    },
}
