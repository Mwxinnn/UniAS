
## Model structure:

## Backbone
### EfficientnetB4

__init__: block数，通道数，相比输入下采样多少倍；\
__forward__: 输出{"res1":[feature], "res2":[feature], ...}, {"res1":[noisy feature], "res2":[noisy feature], ...} \
__out_shape__: {"res{i}": channel[i], stride[i] }

## Pixel decoder 
### Modified U-Net: TransformerEncoderPixelDecoder

__init__: 输入上面的output_shape字典改的list，最高层特征在最前面, mask_dim \
__forward__: 返回self.mask_features(y) --形状为B,24,H/2,W/2 |  transformer_encoder_features --形状为B,160,H/8,W/8 | multi_scale_features List:[y1,y2,y3,y4]维度从高到低

## mask2former_transformer_decoder
### MultiScaleMaskedTransformerDecoder（用了偏低维的三层decoder）

__init__: 输入input_shape \
__forward__: 输入某前3层feature的干净状态和重建效果（高维特征放在前面），返回
```python
out = {'pred_logits': predictions_class[-1], # B Num_cls 最后一层的类别logits 
       'recon_pics': recon_pics[-1], # 与前面特征图的尺寸相对应，最后一层的输出*特征
       'aux_outputs':[{"pred_logits": a, "recon_pics": b}] # 前面两层的特征}
```

## maskformer_transformer_decoder
### StandardTransformerDecoder（只用了中间层feature的decoder）

__init__: 输入input_shape \
__forward__: 最后一层特征的干净和加噪音状态

## Criterion 

label是一个{"cls:[logits],"feature":[[feature_deep],...,[feature_lower],"image":input_image]}