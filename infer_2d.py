#!/usr/bin/env python3
"""
MedSAM2 2D 图像批量推理脚本（全图 Box Prompt）

使用整个图像作为 box prompt，对目录下的所有图像进行分割推理。
输出的 mask 保持原图分辨率。

用法:
    python infer_2d.py \
        --image_dir /path/to/images/ \
        --checkpoint /path/to/medsam2.pt \
        --output_dir /path/to/predictions/

    # 可选参数
    python infer_2d.py \
        --image_dir /path/to/images/ \
        --checkpoint /path/to/medsam2.pt \
        --output_dir /path/to/predictions/ \
        --sam2_cfg configs/sam2.1_hiera_t512.yaml \
        --device cuda \
        --ext .png .jpg
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor_npz


def collect_images(image_dir, exts):
    """收集目录中所有图像文件，按文件名排序。"""
    paths = []
    for name in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            paths.append(os.path.join(image_dir, name))
    return paths


def load_rgb_tensor(image_path):
    """
    加载图像，返回 [3, H, W] tensor，值域 [0, 1]。
    """
    img = Image.open(image_path).convert("RGB")
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return tensor


def preprocess(img_tensor, image_size=512):
    """
    预处理为 MedSAM2 期望的格式。
    输入: [3, H, W], [0, 1]
    输出: [3, image_size, image_size], ImageNet 归一化
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    C, H, W = img_tensor.shape
    if H != image_size or W != image_size:
        img = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    else:
        img = img_tensor.clone()

    return (img - mean) / std


def infer_one(predictor, img_tensor, device):
    """
    对单张图像进行推理（全图 box prompt）。
    输入: img_tensor [3, H, W], [0, 1]
    输出: mask_logits [1, H, W] — 原图尺寸
    """
    H_orig, W_orig = img_tensor.shape[1], img_tensor.shape[2]

    # 预处理并构造单帧 "视频"
    img_processed = preprocess(img_tensor, predictor.image_size).unsqueeze(0).to(device)  # [1, 3, 512, 512]

    # 初始化 inference state
    inference_state = predictor.init_state(
        images=img_processed,
        video_height=H_orig,
        video_width=W_orig,
    )

    # 全图 box prompt
    box = np.array([0, 0, W_orig - 1, H_orig - 1], dtype=np.float32)
    _, _, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        box=box,
    )

    # out_mask_logits: [num_obj, 1, H_orig, W_orig]，已自动映射回原始尺寸
    mask_logits = out_mask_logits[0].float()  # [1, H, W]

    predictor.reset_state(inference_state)
    return mask_logits


def logits_to_binary_mask(logits):
    """将 logits 转为二值 mask。"""
    np_logits = logits.squeeze().cpu().numpy()
    # 由于 logits 可能非常大（-100 ~ 100），clip 防止溢出
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(np_logits, -50, 50)))
    return (sigmoid > 0.5).astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser(
        description="MedSAM2 2D 图像批量推理（全图 Box Prompt）"
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="输入图像目录")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="MedSAM2 权重文件路径 (.pt)")
    parser.add_argument("--output_dir", type=str, default="./predictions",
                        help="输出 mask 目录（默认 ./predictions）")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1_hiera_t512.yaml",
                        help="模型配置 yaml（默认 configs/sam2.1_hiera_t512.yaml）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备（默认 cuda）")
    parser.add_argument("--ext", type=str, nargs="+",
                        default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
                        help="图像扩展名列表")
    args = parser.parse_args()

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # 处理 config 路径（hydra 需要绝对路径或 '//' 前缀）
    if not args.sam2_cfg.startswith("//"):
        if not os.path.isabs(args.sam2_cfg):
            cfg_path = os.path.join(os.getcwd(), "sam2", args.sam2_cfg)
        else:
            cfg_path = args.sam2_cfg
        cfg_path = "//" + cfg_path
    else:
        cfg_path = args.sam2_cfg

    # 加载模型
    print(f"[加载] checkpoint: {args.checkpoint}")
    print(f"[加载] config:    {cfg_path}")
    predictor = build_sam2_video_predictor_npz(
        config_file=cfg_path,
        ckpt_path=args.checkpoint,
        device=device,
        mode="eval",
    )

    # 收集图像
    exts = set(e.lower() for e in args.ext)
    image_paths = collect_images(args.image_dir, exts)
    if not image_paths:
        print(f"[错误] 在 {args.image_dir} 中没有找到匹配的图像文件")
        return
    print(f"[找到] {len(image_paths)} 张图像")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 推理
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.nullcontext():
        for i, img_path in enumerate(image_paths):
            stem = os.path.splitext(os.path.basename(img_path))[0]

            # 加载图像
            img_tensor = load_rgb_tensor(img_path)  # [3, H, W], [0, 1]
            H, W = img_tensor.shape[1], img_tensor.shape[2]

            # 推理
            mask_logits = infer_one(predictor, img_tensor, device)

            # 确保输出尺寸与原图一致
            _, out_H, out_W = mask_logits.shape
            if out_H != H or out_W != W:
                mask_logits = F.interpolate(
                    mask_logits.unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)

            # 转为二值 mask 并保存
            mask_bin = logits_to_binary_mask(mask_logits)
            out_path = os.path.join(args.output_dir, f"{stem}.png")
            Image.fromarray(mask_bin, mode="L").save(out_path)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(image_paths)}] {stem}.png  (原图 {W}×{H})")

    print(f"\n[完成] 共处理 {len(image_paths)} 张图像，输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
