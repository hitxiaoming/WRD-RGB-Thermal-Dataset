import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from mmpretrain.registry import MODELS, DATASETS, TRANSFORMS
from mmpretrain.models.classifiers import ImageClassifier
from mmcv.transforms import BaseTransform
from mmpretrain.datasets import CustomDataset

# -------------------------------------------------------------------
# Dual-branch model
# -------------------------------------------------------------------
@DATASETS.register_module()
class RGBIRPairDataset(CustomDataset):

    def load_data_list(self):
        data_list = super().load_data_list()
        new_data_list = []
        for item in data_list:
            rgb_path = item['img_path']
  
            if 'rgb' in rgb_path:
                ir_path = rgb_path.replace('/rgb/', '/ir/').replace('\\rgb\\', '\\ir\\')
            else:
                ir_path = rgb_path.replace('rgb', 'ir')
            
            item['ir_path'] = ir_path
            new_data_list.append(item)
        return new_data_list

@TRANSFORMS.register_module()
class LoadRGBIRCombined(BaseTransform):

    def transform(self, results):
        rgb_path = results['img_path']
        ir_path = results['ir_path']
        
        try:

            img_rgb = Image.open(rgb_path).convert('RGB')
            img_rgb = np.array(img_rgb)

            img_ir = Image.open(ir_path).convert('L') 
            img_ir = np.array(img_ir)
            img_ir = np.stack([img_ir] * 3, axis=2)   

            if img_rgb.shape[:2] != img_ir.shape[:2]:
                h, w = img_rgb.shape[:2]
                img_ir = cv2.resize(img_ir, (w, h), interpolation=cv2.INTER_LINEAR)
            

            img_cat = np.concatenate((img_rgb, img_ir), axis=2)
            
            results['img'] = img_cat
            results['img_shape'] = img_cat.shape
            results['ori_shape'] = img_cat.shape
            return results
        except Exception as e:
            print(f"Error loading {rgb_path}: {e}")
            raise e

# -------------------------------------------------------------------
# Backbone
# -------------------------------------------------------------------
@MODELS.register_module()
class DualStreamClassifier(ImageClassifier):
    def __init__(self, backbone, neck=None, head=None, pretrained=None, data_preprocessor=None, init_cfg=None):
        super(ImageClassifier, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        

        self.backbone_rgb = MODELS.build(backbone)
        self.backbone_ir = MODELS.build(backbone)

        if neck is not None:
            self.neck_rgb = MODELS.build(neck)
            self.neck_ir = MODELS.build(neck)
        

        if head is not None:
            self.head = MODELS.build(head)

        if pretrained is not None:
            self.init_weights(pretrained)

    def extract_feat(self, inputs):
        # inputs shape: [Batch, 6, H, W]

        x_rgb = inputs[:, 0:3, :, :]
        x_ir = inputs[:, 3:6, :, :]
        

        feat_rgb = self.backbone_rgb(x_rgb) 
        feat_ir = self.backbone_ir(x_ir)

        if hasattr(self, 'neck_rgb'):
            feat_rgb = self.neck_rgb(feat_rgb)
            feat_ir = self.neck_ir(feat_ir)
            

        if isinstance(feat_rgb, tuple): feat_rgb = feat_rgb[0]
        if isinstance(feat_ir, tuple): feat_ir = feat_ir[0]

        # feat_rgb shape: [B, 768], feat_ir shape: [B, 768]
        # output shape: [B, 1536]
        feat_fused = torch.cat([feat_rgb, feat_ir], dim=1)
        
        return (feat_fused, )

    def loss(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)

    def predict(self, inputs, data_samples=None, **kwargs):
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, data_samples, **kwargs)