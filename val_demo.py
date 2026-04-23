import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmpretrain.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings

CONFIG_PATH = r'configs\sp_convnextv2.py'
WEIGHT_PATH = r'weight\sp_convnextv2.pth'
DATA_ROOT = r'data\demo'
RESULT_ROOT = r'data\demo_result'

RGB_DIR = os.path.join(DATA_ROOT, 'rgb')
IR_DIR = os.path.join(DATA_ROOT, 'ir')
RESULT_RGB_DIR = os.path.join(RESULT_ROOT, 'rgb')
RESULT_IR_DIR = os.path.join(RESULT_ROOT, 'ir')
CLASSES = [
    'di_asphalt', 'dry_asphalt', 'dry_concrete', 'dry_sand_gravel',
    'ice', 'snow', 'wet_asphalt', 'wet_concrete'
]
SCALE = 256
CROP_SIZE = 224
MEAN = np.array([123.675, 116.28, 103.53, 123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375, 58.395, 57.12, 57.375], dtype=np.float32)
def process_images(rgb_path, ir_path, device):
    img_rgb = cv2.imread(rgb_path)
    img_ir = cv2.imread(ir_path)
    if img_rgb is None or img_ir is None:
        return None, None, None
    orig_rgb = img_rgb.copy()
    orig_ir = img_ir.copy()
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale_factor = SCALE / min(h, w)
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    img_ir = cv2.resize(img_ir, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    y1 = max(0, int(round((new_h - CROP_SIZE) / 2.0)))
    x1 = max(0, int(round((new_w - CROP_SIZE) / 2.0)))
    img_rgb = img_rgb[y1:y1 + CROP_SIZE, x1:x1 + CROP_SIZE]
    img_ir = img_ir[y1:y1 + CROP_SIZE, x1:x1 + CROP_SIZE]
    img_concat = np.concatenate([img_rgb, img_ir], axis=-1)
    img_concat = (img_concat - MEAN) / STD
    img_tensor = torch.from_numpy(img_concat).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor.to(device), orig_rgb, orig_ir
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading config from {CONFIG_PATH}...")
    cfg = Config.fromfile(CONFIG_PATH)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    print("Building model and loading checkpoint...")
    model = MODELS.build(cfg.model)
    load_checkpoint(model, WEIGHT_PATH, map_location='cpu')
    model.to(device)
    model.eval()
    class_names = [d for d in os.listdir(RGB_DIR) if os.path.isdir(os.path.join(RGB_DIR, d))]
    total_images = 0
    total_correct = 0
    print("\nStarting evaluation...")
    for class_name in class_names:
        class_rgb_dir = os.path.join(RGB_DIR, class_name)
        class_ir_dir = os.path.join(IR_DIR, class_name)

        save_class_rgb_dir = os.path.join(RESULT_RGB_DIR, class_name)
        save_class_ir_dir = os.path.join(RESULT_IR_DIR, class_name)
        os.makedirs(save_class_rgb_dir, exist_ok=True)
        os.makedirs(save_class_ir_dir, exist_ok=True)
        
        image_list = os.listdir(class_rgb_dir)
        print(f"Processing [{class_name}] - {len(image_list)} images...")
        
        for img_name in image_list:
            rgb_path = os.path.join(class_rgb_dir, img_name)
            ir_path = os.path.join(class_ir_dir, img_name)

            if not os.path.exists(ir_path):
                continue

            input_tensor, orig_rgb, orig_ir = process_images(rgb_path, ir_path, device)
            if input_tensor is None:
                continue


            with torch.no_grad():
                logits = model(input_tensor, mode='tensor')
                probs = F.softmax(logits, dim=1)
                
                confidence, pred_idx = torch.max(probs, dim=1)
                confidence = confidence.item()
                pred_idx = pred_idx.item()

            pred_class_name = CLASSES[pred_idx]

            total_images += 1
            is_correct = (pred_class_name == class_name)
            if is_correct:
                total_correct += 1

            text_class = f"{pred_class_name}"
            text_conf = f"Conf: {confidence:.2f}"

            color_pred = (0, 255, 0) if is_correct else (0, 0, 255)

            cv2.putText(orig_rgb, text_class, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pred, 1, cv2.LINE_AA)
            cv2.putText(orig_rgb, text_conf, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pred, 1, cv2.LINE_AA)

            cv2.putText(orig_ir, text_class, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pred, 1, cv2.LINE_AA)
            cv2.putText(orig_ir, text_conf, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pred, 1, cv2.LINE_AA)

            save_rgb_path = os.path.join(save_class_rgb_dir, img_name)
            save_ir_path = os.path.join(save_class_ir_dir, img_name)
            
            cv2.imwrite(save_rgb_path, orig_rgb)
            cv2.imwrite(save_ir_path, orig_ir)

    if total_images > 0:
        accuracy = (total_correct / total_images) * 100
        print(f"\nVerification finished!")
        print(f"Total Images Processed: {total_images}")
        print(f"Total Correct: {total_correct}")
        print(f"Average Accuracy: {accuracy:.2f}%")
        print(f"All visual results are saved in '{RESULT_ROOT}'.")
    else:
        print("\nNo valid images found for evaluation.")

if __name__ == '__main__':
    main()