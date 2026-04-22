import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image


from mmpretrain.registry import MODELS, DATASETS, TRANSFORMS
from mmpretrain.models.backbones import EdgeNeXt  
from mmcv.transforms import BaseTransform
from mmpretrain.datasets import CustomDataset
from mmengine.logging import MMLogger

MODEL_DEFAULTS = {
    'input_mode': 'rgb_ir', 
    'ir2rgb_weights': [0.1, 0.1, 0.1, 0.1],  
    'rgb2ir_weights': [0.1, 0.1, 0.1, 0.1],
    'fix_weights': False,
    'symmetric_interaction': False,
}

AUGMENT_DEFAULTS = {
    'erase_prob': 0.5,
    'min_area_ratio': 0.02,
    'max_area_ratio': 1/3,
    'aspect_range': (0.3, 3.3),
}


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
            img_ir = np.stack([img_ir] * 3, axis=2) # 扩展为3通道
            
            if img_rgb.shape[:2] != img_ir.shape[:2]:
                h, w = img_rgb.shape[:2]
                img_ir = cv2.resize(img_ir, (w, h), interpolation=cv2.INTER_LINEAR)
            
            img_cat = np.concatenate((img_rgb, img_ir), axis=2) # (H, W, 6)
            results['img'] = img_cat
            results['img_shape'] = img_cat.shape
            results['ori_shape'] = img_cat.shape
            return results
        except Exception as e:
            print(f"Error loading {rgb_path}: {e}")
            raise e

@TRANSFORMS.register_module()
class RandomErasingMultiChannel(BaseTransform):
    def __init__(self, 
                 erase_prob=AUGMENT_DEFAULTS['erase_prob'], 
                 min_area_ratio=AUGMENT_DEFAULTS['min_area_ratio'], 
                 max_area_ratio=AUGMENT_DEFAULTS['max_area_ratio'],
                 aspect_range=AUGMENT_DEFAULTS['aspect_range'], 
                 mode='const', fill_color=None, fill_std=None):
        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_range = aspect_range
        self.mode = mode
        self.fill_color = fill_color  
        self.fill_std = fill_std      

    def transform(self, results):
        if np.random.rand() > self.erase_prob:
            return results
        img = results['img'] 
        h, w, c = img.shape
        area = h * w
        for _ in range(10):
            target_area = np.random.uniform(self.min_area_ratio, self.max_area_ratio) * area
            aspect_ratio = np.random.uniform(self.aspect_range[0], self.aspect_range[1])
            h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
            w_erase = int(round(np.sqrt(target_area / aspect_ratio)))
            if w_erase < w and h_erase < h:
                x1 = np.random.randint(0, w - w_erase)
                y1 = np.random.randint(0, h - h_erase)
                if self.mode == 'rand':
                    mean = np.array(self.fill_color) if self.fill_color is not None else np.zeros(c)
                    std = np.array(self.fill_std) if self.fill_std is not None else np.ones(c)
                    noise = np.random.randn(h_erase, w_erase, c)
                    img[y1:y1+h_erase, x1:x1+w_erase, :] = noise * std.reshape(1,1,c) + mean.reshape(1,1,c)
                else:
                    fill_val = np.array(self.fill_color) if self.fill_color is not None else np.zeros(c)
                    img[y1:y1+h_erase, x1:x1+w_erase, :] = fill_val.reshape(1,1,c)
                results['img'] = img
                return results
        return results


@MODELS.register_module()
class DyRoadNet(EdgeNeXt): 
    """
    DyRoadNet: Dual-stream architecture using mmpretrain's EdgeNeXt.
    """
    _config_logged = False 

    def __init__(self, 
                 input_mode=MODEL_DEFAULTS['input_mode'],
                 ir2rgb_weights=MODEL_DEFAULTS['ir2rgb_weights'],
                 rgb2ir_weights=MODEL_DEFAULTS['rgb2ir_weights'],
                 fix_weights=MODEL_DEFAULTS['fix_weights'],
                 symmetric_interaction=MODEL_DEFAULTS['symmetric_interaction'], 
                 **kwargs):
        
  
        if 'arch' not in kwargs:
            kwargs['arch'] = 'small'

        super().__init__(**kwargs)
        
        self.input_mode = input_mode.lower()
        

        if not DyRoadNet._config_logged:
            logger = MMLogger.get_current_instance()
            printer = logger.info if logger else print
            printer(f"\n{'='*40}")
            printer(f" [DyRoadNet] Configuration (MMPretrain EdgeNeXt)")
            printer(f"{'='*40}")
            printer(f" Backbone Arch:       {kwargs['arch']}")
            printer(f" Input Mode:          {self.input_mode.upper()}")
            printer(f" Fix Weights:         {fix_weights}")
            printer(f" Symmetric Mode:      {symmetric_interaction}")
            printer(f" IR->RGB Init:        {ir2rgb_weights}")
            printer(f"{'='*40}\n")
            DyRoadNet._config_logged = True

        self.symmetric_interaction = symmetric_interaction
        self.fix_weights = fix_weights
        

        num_stages = len(self.stages) 


        def create_param_list(values):
            param_list = nn.ParameterList()
            for i in range(num_stages):
                val = float(values[i]) if i < len(values) else float(values[-1])
                p = nn.Parameter(torch.full((1,), val))
                if self.fix_weights or val == 0.0:
                    p.requires_grad = False
                param_list.append(p)
            return param_list

        if self.symmetric_interaction:
            self.inter_weights = create_param_list(ir2rgb_weights)
        else:
            self.ir_to_rgb_weights = create_param_list(ir2rgb_weights)
            self.rgb_to_ir_weights = create_param_list(rgb2ir_weights)

    def forward(self, x):


        if self.input_mode == 'rgb_ir':
            x_rgb = x[:, :3, :, :]
            x_ir = x[:, 3:, :, :]
        elif self.input_mode == 'rgb_rgb':
            x_rgb = x[:, :3, :, :]
            x_ir = x[:, :3, :, :] 
        elif self.input_mode == 'ir_ir':
            x_rgb = x[:, 3:, :, :]
            x_ir = x[:, 3:, :, :]
        else:
            x_rgb = x[:, :3, :, :]
            x_ir = x[:, 3:, :, :]

        outs = []

        for i, stage in enumerate(self.stages):

            x_rgb = self.downsample_layers[i](x_rgb) 

            x_rgb = stage(x_rgb)

            if self.pos_embed is not None and i == 0:
                B, _, H, W = x_rgb.shape

                x_rgb = x_rgb + self.pos_embed(x_rgb)


            x_ir = self.downsample_layers[i](x_ir)
            x_ir = stage(x_ir)
            if self.pos_embed is not None and i == 0:
                x_ir = x_ir + self.pos_embed(x_ir)


            if self.symmetric_interaction:
                alpha = self.inter_weights[i]
                tmp_rgb = x_rgb + alpha * x_ir
                tmp_ir  = x_ir  + alpha * x_rgb
            else:
                alpha_ir2rgb = self.ir_to_rgb_weights[i]
                alpha_rgb2ir = self.rgb_to_ir_weights[i]
                tmp_rgb = x_rgb + alpha_ir2rgb * x_ir
                tmp_ir  = x_ir  + alpha_rgb2ir * x_rgb
            
            x_rgb, x_ir = tmp_rgb, tmp_ir


            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    # GAP -> Norm -> Concat
                    gap_rgb = x_rgb.mean([-2, -1], keepdim=True)
                    gap_ir  = x_ir.mean([-2, -1], keepdim=True)
                    feat = torch.cat([norm_layer(gap_rgb.flatten(1)), 
                                      norm_layer(gap_ir.flatten(1))], dim=1)
                    outs.append(feat)
                else:

                    feat_rgb = norm_layer(x_rgb).contiguous()
                    feat_ir  = norm_layer(x_ir).contiguous()
                    feat = torch.cat([feat_rgb, feat_ir], dim=1)
                    outs.append(feat)

        return tuple(outs)

    def print_interaction_weights(self):

        logger = MMLogger.get_current_instance()
        mode_str = 'Symmetric' if self.symmetric_interaction else 'Asymmetric'
        fix_str = "(FIXED)" if self.fix_weights else "(LEARNABLE)"
        
        logger.info(f"--- DyRoadNet [{self.input_mode.upper()}] Weights {mode_str} {fix_str} ---")
        for i in range(len(self.stages)):
            if self.symmetric_interaction:
                v = self.inter_weights[i].item()
                logger.info(f"Stage {i}: Interaction Strength = {v:.4f}")
            else:
                v1 = self.ir_to_rgb_weights[i].item()
                v2 = self.rgb_to_ir_weights[i].item()
                logger.info(f"Stage {i}: IR->RGB={v1:.4f} | RGB->IR={v2:.4f}")