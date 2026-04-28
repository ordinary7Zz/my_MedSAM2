import torch
import torch.nn.functional as F
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from dino_unet import DINOv3_S_UNet

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
        
        # 加载MedSAM2模型
        self.sam2_model = build_sam2(
            config_file=sam2_cfg,
            checkpoint_path=checkpoint_path,
            device=self.device,
            mode="eval"
        )
        
        # 创建图像预测器
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        # 加载DINO-UNet模型（如果提供了检查点）
        self.dino_unet_model = None
        if dino_unet_ckpt:
            self.dino_unet_model = DINOv3_S_UNet()
            self.dino_unet_model.load_state_dict(torch.load(dino_unet_ckpt, map_location=self.device))
            self.dino_unet_model.to(self.device)
            self.dino_unet_model.eval()
    
    def __call__(self, image, batch=None):
        with torch.no_grad():
            batch_size = image.shape[0]
            results = []
            
            for i in range(batch_size):
                img_tensor = batch["image_rgb"][i]
                img_processed = batch["image"][i]
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                print(f"[Debug] img_np min: {img_np.min()}, max: {img_np.max()}")
                H, W = img_np.shape[:2]
                self.predictor.set_image(img_np)

                # 使用DINO-UNet预测前景区域，然后从中提取前景和背景点
                point_coords = None
                point_labels = None
                label_map = batch['label'][i][0].cpu().numpy()

                if self.dino_unet_model is not None:
                    dino_input = img_processed.unsqueeze(0).to(self.device)
                    dino_logits = self.dino_unet_model(dino_input).squeeze(0).squeeze(0)
                    dino_prob = torch.sigmoid(dino_logits).detach().cpu().numpy().astype(np.float32)
                    # === 二值化对比分析 ===
                    dino_bin = (dino_prob > 0.5).astype(np.uint8)
                    label_bin = (label_map > 0.5).astype(np.uint8)
                    intersection = np.logical_and(dino_bin, label_bin).sum()
                    union = np.logical_or(dino_bin, label_bin).sum()
                    dice = 2 * intersection / (dino_bin.sum() + label_bin.sum() + 1e-6)
                    print(f"[DINO vs Label] Sample {i}: Dice={dice:.4f}, DINO sum={dino_bin.sum()}, Label sum={label_bin.sum()}, Intersection={intersection}, Union={union}")

                    # 提取前景点（从概率大于0.7的区域均匀选择）
                    ys_foreground, xs_foreground = np.where(dino_prob > 0.5)
                    foreground_points = []
                    if len(xs_foreground) > 0:
                        num_foreground = min(5, len(xs_foreground))
                        foreground_points = sample_uniformly(ys_foreground, xs_foreground, num_foreground)
                    
                    # 提取背景点（从概率小于0.3的区域均匀选择）
                    ys_background, xs_background = np.where(dino_prob < 0.5)
                    background_points = []
                    if len(xs_background) > 0:
                        num_background = min(3, len(xs_background))
                        background_points = sample_uniformly(ys_background, xs_background, num_background)
                    
                    # 合并前景和背景点
                    if foreground_points or background_points:
                        all_points = foreground_points + background_points
                        point_coords = np.array(all_points, dtype=np.float32)
                        point_labels = np.array([1]*len(foreground_points) + [0]*len(background_points), dtype=np.int32)
                
                # 如果没有足够的点，使用图像中心作为前景点，四角作为背景点
                if point_coords is None or len(point_coords) < 1:
                    center_x, center_y = W // 2, H // 2
                    point_coords = np.array([[center_x, center_y]], dtype=np.float32)
                    point_labels = np.array([1], dtype=np.int32)

                # 使用前景和背景点进行推理
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                    return_logits=False,
                    normalize_coords=True
                )

                best_mask = masks[np.argmax(scores)]
                print(f"[Debug] best_mask min: {best_mask.min()}, max: {best_mask.max()}, unique values: {np.unique(best_mask)}")
                
                # === best_mask与label_map二值化对比分析 ===
                best_mask_bin = (best_mask).astype(np.uint8)
                label_bin = (label_map).astype(np.uint8)

                # 保存best_mask_bin和label_bin到plot目录
                import os
                from PIL import Image
                plot_dir = "plot"
                os.makedirs(plot_dir, exist_ok=True)
                # 使用时间戳区分文件名
                import time
                ts = int(time.time())
                Image.fromarray(img_np).save("debug_img.png")
                Image.fromarray((dino_bin*255).astype(np.uint8)).save("debug_dino.png")
                Image.fromarray((best_mask_bin*255).astype(np.uint8)).save("debug_sam.png")
                Image.fromarray((label_bin*255).astype(np.uint8)).save("debug_gt.png")

                intersection2 = np.logical_and(best_mask, label_bin).sum()
                union2 = np.logical_or(best_mask, label_bin).sum()
                dice2 = 2 * intersection2 / (best_mask.sum() + label_bin.sum() + 1e-6)
                print(f"[MedSAM2 vs Label] Sample {i}: Dice={dice2:.4f}, MedSAM2 sum={best_mask.sum()}, Label sum={label_bin.sum()}, Intersection={intersection2}, Union={union2}")

                mask_tensor = torch.from_numpy(best_mask).float().unsqueeze(0).to(self.device)
                results.append(mask_tensor)
            
            # 合并结果批次
            mask_pred = torch.stack(results, dim=0)
            
            return mask_pred
    
    def eval(self):
        """设置模型为评估模式（兼容torch.nn.Module接口）"""
        if hasattr(self, 'sam2_model'):
            self.sam2_model.eval()
        if hasattr(self, 'dino_unet_model') and self.dino_unet_model is not None:
            self.dino_unet_model.eval()
        return self
    
    def to(self, device):
        """将模型移动到指定设备（兼容torch.nn.Module接口）"""
        self.device = device
        if hasattr(self, 'sam2_model'):
            self.sam2_model = self.sam2_model.to(device)
            self.predictor = SAM2ImagePredictor(self.sam2_model)
        if hasattr(self, 'dino_unet_model') and self.dino_unet_model is not None:
            self.dino_unet_model = self.dino_unet_model.to(device)
        return self
