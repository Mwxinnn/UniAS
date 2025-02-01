# TODO
from modeling.meta_arch.maskformer_head import MaskFormerHead



def build_seg_head(cfg, input_shape):
    return MaskFormerHead(cfg, input_shape)
