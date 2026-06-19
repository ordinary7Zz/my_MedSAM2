"""
快速调试脚本：检查 SAM2 预测的 mask logits 范围，
以及 GT mask 是否被正确加载（排除反转问题）。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def check_gt_mask(gt_path, num_samples=5):
    """检查 GT mask 是否正常（前景应该是白色/255）"""
    from dataset import _collect_files
    files = _collect_files(gt_path)[:num_samples]
    print(f"\n{'='*60}")
    print(f"Checking GT masks from: {gt_path}")
    print(f"{'='*60}")
    for f in files:
        img = Image.open(f).convert('L')
        arr = np.array(img)
        total = arr.size
        fg_255 = (arr > 128).sum()
        fg_ratio = fg_255 / total
        print(f"  {os.path.basename(f)}: shape={arr.shape}, "
              f"min={arr.min()}, max={arr.max()}, "
              f"fg_pixels(>128)={fg_255}, fg_ratio={fg_ratio:.3f}")
        if fg_ratio > 0.5:
            print(f"    ⚠️  WARNING: Foreground > 50%! Mask might be INVERTED!")


def check_sam2_prediction(sam2_cfg, checkpoint, image_path, gt_path, device="cuda"):
    """检查 SAM2 对一张图像的预测 logits 范围"""
    from dataset import _collect_files
    
    print(f"\n{'='*60}")
    print(f"Checking SAM2 prediction")
    print(f"{'='*60}")
    
    # 加载模型
    model = build_sam2(config_file=sam2_cfg, checkpoint_path=checkpoint, device=device, mode="eval")
    predictor = SAM2ImagePredictor(model)
    
    # 加载一张图片
    img_files = _collect_files(image_path)[:3]
    gt_files = _collect_files(gt_path)[:3]
    
    for img_f, gt_f in zip(img_files, gt_files):
        # 加载图像
        img = Image.open(img_f).convert('RGB')
        img_resized = img.resize((512, 512), Image.BILINEAR)
        img_np = np.array(img_resized)  # (512, 512, 3), uint8
        
        # 加载 GT
        gt = Image.open(gt_f).convert('L')
        gt_resized = gt.resize((512, 512), Image.NEAREST)
        gt_np = (np.array(gt_resized) > 128).astype(np.uint8)
        
        # 计算 box
        if gt_np.sum() == 0:
            print(f"  {os.path.basename(img_f)}: GT is empty, skipping")
            continue
        ys, xs = np.where(gt_np > 0)
        x0, y0, x1, y1 = xs.min()-4, ys.min()-4, xs.max()+4, ys.max()+4
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(511, x1); y1 = min(511, y1)
        box = np.array([[x0, y0, x1, y1]], dtype=np.float32)
        
        # 设置图像
        predictor.set_image(img_np)
        
        # 预测 logits
        masks_logits, scores, _ = predictor.predict(
            box=box,
            multimask_output=False,
            return_logits=True,
            normalize_coords=True,
        )
        
        # 预测 binary
        masks_binary, scores_b, _ = predictor.predict(
            box=box,
            multimask_output=False,
            return_logits=False,
            normalize_coords=True,
        )
        
        logits = masks_logits[0]  # (H, W)
        binary = masks_binary[0]  # (H, W) bool
        
        # 统计
        print(f"\n  Image: {os.path.basename(img_f)}")
        print(f"    GT: fg_pixels={gt_np.sum()}, fg_ratio={gt_np.sum()/gt_np.size:.3f}")
        print(f"    Box: [{x0}, {y0}, {x1}, {y1}]")
        print(f"    Logits: min={logits.min():.4f}, max={logits.max():.4f}, "
              f"mean={logits.mean():.4f}")
        print(f"    Logits > 0: {(logits > 0).sum()} pixels ({(logits > 0).sum()/logits.size*100:.1f}%)")
        print(f"    Binary mask: fg_pixels={binary.sum()}, fg_ratio={binary.sum()/binary.size:.3f}")
        print(f"    Score: {scores[0]:.4f}")
        
        # 计算 Dice
        pred_bin = (logits > 0).astype(np.float32)
        intersection = (pred_bin * gt_np).sum()
        dice = (2 * intersection + 1) / (pred_bin.sum() + gt_np.sum() + 1)
        print(f"    Dice (logits>0 vs GT): {dice:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # 先检查 GT mask 是否反转
    check_gt_mask(args.gt_path, num_samples=5)
    
    # 再检查 SAM2 预测
    check_sam2_prediction(args.sam2_cfg, args.checkpoint, args.image_path, args.gt_path, args.device)
