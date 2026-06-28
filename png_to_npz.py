#!/usr/bin/env python3
"""
将 PNG 格式的 2D 甲状腺超声图像和 mask 转换为 MedSAM2 训练所需的 .npz 格式。

方案 A：每张 2D 图像生成一个独立的 .npz 文件（D=1，即单帧"视频"）。
适用于配置文件中 num_frames=1 的 2D 微调场景。

用法:
    python png_to_npz.py \
        --image_dir /path/to/images/ \
        --mask_dir /path/to/masks/ \
        --output_dir /path/to/train_npz/

目录结构要求:
    image_dir/                  mask_dir/
    ├── case_001.png            ├── case_001.png      ← stem 必须与图像一致
    ├── case_002.jpg            ├── case_002.png
    └── ...                     └── ...

输出 .npz 文件结构:
    imgs:  shape (1, H, W), dtype uint8, 值域 [0, 255]
    gts:   shape (1, H, W), dtype uint8, 0=背景, 1=病灶
"""

import os
import argparse
import numpy as np
from PIL import Image


def find_pairs(image_dir, mask_dir):
    """
    按文件名 stem 匹配图像和 mask，返回 (img_path, mask_path, stem) 列表。
    支持 .jpg/.jpeg/.png/.bmp/.tif/.tiff/.webp 等常见图像格式。
    """
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp',
            '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF', '.WEBP'}

    image_map = {}
    for name in sorted(os.listdir(image_dir)):
        stem, ext = os.path.splitext(name)
        if ext in exts:
            image_map[stem] = os.path.join(image_dir, name)

    mask_map = {}
    for name in sorted(os.listdir(mask_dir)):
        stem, ext = os.path.splitext(name)
        if ext in exts:
            mask_map[stem] = os.path.join(mask_dir, name)

    common_stems = sorted(set(image_map.keys()) & set(mask_map.keys()))
    pairs = [(image_map[s], mask_map[s], s) for s in common_stems]
    return pairs


def convert_single(img_path, mask_path, stem, output_dir):
    """
    将单对 image + mask 转为 .npz 文件。
    """
    # 读取灰度图像
    img = np.array(Image.open(img_path).convert("L"))       # (H, W), uint8

    # 读取 mask 并二值化
    mask = np.array(Image.open(mask_path).convert("L"))     # (H, W)
    mask = (mask > 128).astype(np.uint8)                     # 前景 → 1, 背景 → 0

    # 扩展为 (1, H, W)
    imgs = img[np.newaxis, :, :]
    gts = mask[np.newaxis, :, :]

    # 保存
    output_path = os.path.join(output_dir, f"{stem}.npz")
    np.savez_compressed(output_path, imgs=imgs, gts=gts)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="将 2D PNG 图像和 mask 转换为 MedSAM2 训练 NPZ 格式"
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="原始超声图像目录")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="GT mask 目录（二值图，前景≠0）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出 .npz 文件目录")
    parser.add_argument("--mask_threshold", type=int, default=128,
                        help="mask 二值化阈值，默认 128")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 匹配图像-mask pair
    pairs = find_pairs(args.image_dir, args.mask_dir)
    if not pairs:
        print(f"[错误] 在 {args.image_dir} 和 {args.mask_dir} 中没有找到匹配的图像-mask pair")
        print(f"       请确保两个目录下的文件 stem 一致")
        return

    print(f"找到 {len(pairs)} 对图像-mask pair，开始转换...")

    # 逐对转换
    skipped = 0
    for i, (img_path, mask_path, stem) in enumerate(pairs):
        try:
            out_path = convert_single(img_path, mask_path, stem, args.output_dir)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(pairs)}] {stem}.npz  "
                      f"(图像: {os.path.basename(img_path)}, "
                      f"mask: {os.path.basename(mask_path)})")
        except Exception as e:
            print(f"  [跳过] {stem}: {e}")
            skipped += 1

    print(f"\n转换完成: 成功 {len(pairs) - skipped}/{len(pairs)} 个，跳过 {skipped} 个")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
