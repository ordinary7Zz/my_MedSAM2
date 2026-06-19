"""
==============================================================================
MedSAM2 分割推理 Wrapper（使用 Video Predictor 进行单帧推理）
==============================================================================

【运行流程概述】

本文件实现了 MedSAM2SegWrapper 类，用于对 2D 医学图像进行分割推理。
核心思路：将单张 2D 图像当作只有 1 帧的"视频"，使用 MedSAM2 的 Video Predictor
进行推理。这是因为 MedSAM2 模型是作为 video model 训练的，必须使用 video predictor
的完整 pipeline（包含 memory attention 等机制）才能获得有意义的预测结果。

【推理步骤】

1. 加载模型：
   - 使用 build_sam2_video_predictor_npz 加载 MedSAM2 模型权重
   - 配置路径通过 hydra 解析（需要绝对路径或 '//' 前缀）

2. 图像预处理：
   - 输入图像为 [0,1] 范围的 RGB tensor，shape [3, H, W]
   - Resize 到模型期望的分辨率（通常 512×512）
   - 使用 ImageNet 均值/标准差进行归一化

3. 生成 Box Prompt：
   - 从 GT mask 中提取前景像素的 bounding box
   - 对 box 进行固定 padding（4像素）
   - 【随机扩大】对 box 的每条边进行随机扩大（0~5%的目标尺寸），
     模拟实际使用中 box 不完美贴合目标的情况，增强模型鲁棒性

4. 推理：
   - 初始化 inference state（单帧作为1帧视频）
   - 通过 add_new_points_or_box 传入 box prompt
   - 获取 mask logits 输出

5. 后处理：
   - 返回 raw logits（后续 metrics.py 中做 sigmoid + 阈值化）

6. 调试（可选）：
   - 前几个样本保存 4 列对比图：原图、GT mask、GT+box 叠加、SAM2 预测
   - 打印 logits 的值域范围信息

==============================================================================
"""

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from sam2.build_sam import build_sam2_video_predictor_npz

def sample_uniformly(y_coords, x_coords, num_samples):
    """在高置信度区域内均匀采样点"""
    # 确保输入是1维数组
    # 处理可能的tuple输入（来自np.where的返回值）
    if isinstance(y_coords, tuple):
        y_coords = y_coords[0] if y_coords else np.array([], dtype=np.int32)
    if isinstance(x_coords, tuple):
        x_coords = x_coords[0] if x_coords else np.array([], dtype=np.int32)
    
    # 确保是numpy数组并展平为1维
    y_coords = np.asarray(y_coords).flatten()
    x_coords = np.asarray(x_coords).flatten()
    
    if len(y_coords) < num_samples:
        return list(zip(x_coords, y_coords))
    
    points = np.column_stack((x_coords, y_coords))
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    
    grid_cols = int(np.ceil(np.sqrt(num_samples)))
    grid_rows = int(np.ceil(num_samples / grid_cols))
    x_step = (x_max - x_min) / grid_cols if grid_cols > 1 else 1
    y_step = (y_max - y_min) / grid_rows if grid_rows > 1 else 1
    
    grid_points = {}
    for x, y in points:
        grid_x = int((x - x_min) / x_step) if grid_cols > 1 else 0
        grid_y = int((y - y_min) / y_step) if grid_rows > 1 else 0
        grid_key = (grid_x, grid_y)
        if grid_key not in grid_points:
            grid_points[grid_key] = []
        grid_points[grid_key].append((x, y))
    
    if len(grid_points) > num_samples:
        # 解决np.random.choice不能直接处理元组的问题
        grid_keys = list(grid_points.keys())
        # 使用np.arange确保是1维numpy数组
        selected_indices = np.random.choice(np.arange(len(grid_keys)), num_samples, replace=False)
        selected_grids = [grid_keys[i] for i in selected_indices]
        grid_points = {k: grid_points[k] for k in selected_grids}
    
    sampled_points = []
    while len(sampled_points) < num_samples and grid_points:
        for grid_key in list(grid_points.keys()):
            if grid_points[grid_key]:
                # 使用Python的random.randint代替np.random.choice，避免维度问题
                import random
                point_idx = random.randint(0, len(grid_points[grid_key]) - 1)
                sampled_points.append(grid_points[grid_key].pop(point_idx))
                if len(sampled_points) == num_samples:
                    break
            else:
                del grid_points[grid_key]
    
    return sampled_points[:num_samples]


def _mask_to_box(mask_bin: np.ndarray, pad: int = 4) -> Optional[np.ndarray]:
    """
    从二值 mask 中提取 bounding box。
    
    参数:
        mask_bin: 二值 mask，shape [H, W]，前景为1，背景为0
        pad: 固定 padding 像素数（在随机扩大之前先加的基础 padding）
    
    返回:
        box: [x0, y0, x1, y1] 格式的 numpy 数组，或 None（如果 mask 为空）
    """
    if mask_bin.sum() == 0:
        return None
    ys, xs = np.where(mask_bin > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    H, W = mask_bin.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _random_expand_box(box: np.ndarray, img_h: int, img_w: int,
                       max_expand_ratio: float = 0.2) -> np.ndarray:
    """
    对 bounding box 进行随机扩大。
    
    模拟实际使用场景中，用户/检测器提供的 box 不会完美贴合目标边界的情况。
    每条边独立进行随机扩大，扩大量为目标宽/高的 0~max_expand_ratio 倍。
    
    参数:
        box: [x0, y0, x1, y1] 格式的 numpy 数组
        img_h: 图像高度（用于 clamp 边界）
        img_w: 图像宽度（用于 clamp 边界）
        max_expand_ratio: 最大扩大比例，默认 0.05（即每条边最多扩大目标尺寸的 5%）
    
    返回:
        expanded_box: 随机扩大后的 [x0, y0, x1, y1] numpy 数组
    """
    x0, y0, x1, y1 = box
    box_w = x1 - x0
    box_h = y1 - y0

    # 每条边独立随机扩大 [0, max_expand_ratio * 对应方向尺寸]
    expand_left = random.uniform(0, max_expand_ratio) * box_w
    expand_right = random.uniform(0, max_expand_ratio) * box_w
    expand_top = random.uniform(0, max_expand_ratio) * box_h
    expand_bottom = random.uniform(0, max_expand_ratio) * box_h

    # 扩大 box 并 clamp 到图像边界内
    new_x0 = max(0, x0 - expand_left)
    new_y0 = max(0, y0 - expand_top)
    new_x1 = min(img_w - 1, x1 + expand_right)
    new_y1 = min(img_h - 1, y1 + expand_bottom)

    return np.array([new_x0, new_y0, new_x1, new_y1], dtype=np.float32)


class MedSAM2SegWrapper:
    def __init__(
        self,
        sam2_cfg=None,
        checkpoint_path=None,
        device=None,
        dino_unet_ckpt=None,
    ):
        # 设置设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用 video predictor 加载 MedSAM2 模型
        # MedSAM2 是作为 video model 训练的，必须使用 video predictor 推理
        # 处理 config 路径：hydra 需要绝对路径或带 '//' 前缀的路径
        if sam2_cfg and not sam2_cfg.startswith('//'):
            # 如果是相对路径，转换为 hydra 可识别的绝对路径格式
            import os as _os
            script_dir = _os.path.dirname(_os.path.abspath(__file__))
            # 向上一级到项目根目录，再拼接 sam2/ 前缀
            project_root = _os.path.dirname(script_dir)
            cfg_abs = _os.path.join(project_root, 'sam2', sam2_cfg)
            if _os.path.exists(cfg_abs):
                sam2_cfg = '//' + cfg_abs
            else:
                # 尝试直接在 sam2/configs 中查找
                cfg_abs2 = _os.path.join(project_root, 'sam2', 'configs', sam2_cfg)
                if _os.path.exists(cfg_abs2):
                    sam2_cfg = '//' + cfg_abs2

        self.predictor = build_sam2_video_predictor_npz(
            config_file=sam2_cfg,
            ckpt_path=checkpoint_path,
            device=self.device,
            mode="eval"
        )
        self.image_size = self.predictor.image_size  # 通常为 512

        # 保留参数以兼容旧调用
        self.dino_unet_model = None
    
    def _preprocess_image(self, img_rgb_tensor):
        """
        将 [0,1] RGB tensor 预处理为 MedSAM2 video predictor 期望的格式。
        输入: img_rgb_tensor shape [3, H, W], 值域 [0,1]
        输出: normalized tensor shape [3, image_size, image_size]
        """
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        # Resize to model's image_size if needed
        C, H, W = img_rgb_tensor.shape
        if H != self.image_size or W != self.image_size:
            img = F.interpolate(
                img_rgb_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            img = img_rgb_tensor.clone()

        # Normalize
        img = (img - img_mean) / img_std
        return img

    def __call__(self, image, batch=None):
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.nullcontext()
        with torch.inference_mode(), autocast_ctx:
            batch_size = image.shape[0]
            results = []

            for i in range(batch_size):
                img_tensor = batch["image_rgb"][i]  # [3, H, W], [0, 1]
                H_orig, W_orig = img_tensor.shape[1], img_tensor.shape[2]
                sample_name = None
                if batch is not None and "name" in batch:
                    name_value = batch["name"][i]
                    sample_name = name_value if isinstance(name_value, str) else str(name_value)

                # 预处理图像为 video predictor 期望的格式
                img_preprocessed = self._preprocess_image(img_tensor)  # [3, 512, 512]
                # video predictor 期望 images shape: [num_frames, 3, H, W]
                images = img_preprocessed.unsqueeze(0).to(self.device)  # [1, 3, 512, 512]

                # 初始化 inference state（单帧视为1帧视频）
                inference_state = self.predictor.init_state(
                    images=images,
                    video_height=H_orig,
                    video_width=W_orig,
                )

                # 使用 GT mask 转换为 box prompt
                label_map = batch['label'][i][0].cpu().numpy()
                label_bin = (label_map > 0.5).astype(np.uint8)
                box = _mask_to_box(label_bin, pad=4)

                # 随机扩大 box（模拟实际使用中 box 不完美贴合目标的情况）
                if box is not None:
                    box = _random_expand_box(box, H_orig, W_orig, max_expand_ratio=0.05)

                # 使用 video predictor 的 add_new_points_or_box 推理
                if box is not None:
                    _, _, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=1,
                        box=box,  # [x0, y0, x1, y1] 原始像素坐标
                        normalize_coords=True,
                    )
                else:
                    # fallback: 使用中心点
                    center_x, center_y = W_orig // 2, H_orig // 2
                    points = np.array([[center_x, center_y]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)
                    _, _, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=1,
                        points=points,
                        labels=labels,
                        normalize_coords=True,
                    )

                # out_mask_logits shape: [num_obj, 1, H_orig, W_orig]
                # 取第一个 object 的 mask logits
                mask_logits = out_mask_logits[0]  # [1, H_orig, W_orig]

                # # ---- 调试：前5个样本保存对比图 ----
                # if i < 5:
                #     mask_np = mask_logits.squeeze().cpu().numpy()
                #     img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                #     print(f"[Debug] sample={sample_name} logits range: "
                #           f"min={mask_np.min():.4f}, max={mask_np.max():.4f}, "
                #           f"mean={mask_np.mean():.4f}, "
                #           f"(logits>0) sum={int((mask_np > 0).sum())}")
                #     self._save_debug_comparison(
                #         img_np, label_bin, mask_np, box, sample_name, i
                #     )

                # 返回 logits（metrics.py 中会做 sigmoid + threshold）
                mask_tensor = mask_logits.float().to(self.device)  # [1, H, W]
                results.append(mask_tensor)

                # 重置 predictor state
                self.predictor.reset_state(inference_state)

            # 合并结果批次
            mask_pred = torch.stack(results, dim=0)  # [B, 1, H, W]

            return mask_pred
    
    def _save_debug_comparison(self, img_np, label_bin, pred_mask, box, sample_name, idx):
        """保存调试对比图：原图+GT mask+box 叠加 vs SAM2 预测 mask"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        debug_dir = "./debug_box_prompt"
        os.makedirs(debug_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 1) 原始图像
        axes[0].imshow(img_np)
        axes[0].set_title(f"Input Image\n{sample_name}")
        axes[0].axis('off')

        # 2) GT mask（二值）
        axes[1].imshow(label_bin, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"GT Mask\nsum={label_bin.sum()}")
        axes[1].axis('off')

        # 3) 原图 + GT mask 叠加 + box
        axes[2].imshow(img_np)
        # 半透明 GT mask 叠加
        gt_overlay = np.zeros((*label_bin.shape, 4))
        gt_overlay[label_bin > 0] = [0, 1, 0, 0.4]  # 绿色半透明
        axes[2].imshow(gt_overlay)
        # 绘制 box
        if box is not None:
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[2].add_patch(rect)
            axes[2].set_title(f"GT+Box\nbox=[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]")
        else:
            axes[2].set_title("GT+Box (no box, fallback)")
        axes[2].axis('off')

        # 4) SAM2 预测结果（注意：return_logits=True 时输出是 logits，需 sigmoid 后再阈值化）
        if pred_mask.dtype == bool:
            pred_bin = pred_mask.astype(np.uint8)
        else:
            # 对 logits 做 sigmoid 再阈值化
            pred_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(pred_mask, -50, 50)))
            pred_bin = (pred_sigmoid > 0.5).astype(np.uint8)
        axes[3].imshow(img_np)
        pred_overlay = np.zeros((*pred_bin.shape, 4))
        pred_overlay[pred_bin > 0] = [1, 0, 0, 0.4]  # 红色半透明
        axes[3].imshow(pred_overlay)
        axes[3].set_title(f"SAM2 Pred\nsum={pred_bin.sum()}")
        axes[3].axis('off')

        plt.tight_layout()
        fname = f"debug_{idx}_{sample_name}.png"
        plt.savefig(os.path.join(debug_dir, fname), dpi=100, bbox_inches='tight')
        plt.close(fig)
        # print(f"[Debug] Saved: {os.path.join(debug_dir, fname)}")

    def eval(self):
        """设置模型为评估模式（兼容torch.nn.Module接口）"""
        if hasattr(self, 'predictor'):
            self.predictor.eval()
        return self
    
    def to(self, device):
        """将模型移动到指定设备（兼容torch.nn.Module接口）"""
        self.device = device
        if hasattr(self, 'predictor'):
            self.predictor = self.predictor.to(device)
        return self
