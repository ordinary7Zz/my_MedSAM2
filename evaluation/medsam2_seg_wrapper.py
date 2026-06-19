import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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

        # 保留参数以兼容旧调用，但不再加载 DINO-UNet
        self.dino_unet_model = None
    
    def __call__(self, image, batch=None):
        with torch.no_grad():
            batch_size = image.shape[0]
            results = []

            for i in range(batch_size):
                img_tensor = batch["image_rgb"][i]
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                H, W = img_np.shape[:2]
                sample_name = None
                if batch is not None and "name" in batch:
                    name_value = batch["name"][i]
                    sample_name = name_value if isinstance(name_value, str) else str(name_value)

                self.predictor.set_image(img_np)

                # 使用 GT mask 转换为 box prompt
                label_map = batch['label'][i][0].cpu().numpy()
                label_bin = (label_map > 0.5).astype(np.uint8)
                label_sum = int(label_bin.sum())
                box = _mask_to_box(label_bin, pad=4)

                # 调试输出：观察 mask 是否为空、box 是否合理
                if i < 3 or box is None or (i % 200 == 0):
                    box_str = None if box is None else box.tolist()
                    print(
                        f"[PromptDebug] idx={i} name={sample_name} "
                        f"label_sum={label_sum} box={box_str} "
                        f"img_shape=({H},{W}) fallback={box is None}"
                    )

                # 如果 mask 为空，就退回到中心点提示
                point_coords = None
                point_labels = None
                if box is None:
                    center_x, center_y = W // 2, H // 2
                    point_coords = np.array([[center_x, center_y]], dtype=np.float32)
                    point_labels = np.array([1], dtype=np.int32)

                # 使用 box prompt 进行推理（fallback 为中心点）
                # 注意：使用 return_logits=True 以返回 logits，与 metrics.py 中的 sigmoid 配合
                predict_kwargs = dict(
                    multimask_output=False,
                    return_logits=True,
                    normalize_coords=True,
                )
                if box is not None:
                    predict_kwargs["box"] = box[None, :].astype(np.float32)
                else:
                    predict_kwargs["point_coords"] = point_coords
                    predict_kwargs["point_labels"] = point_labels

                masks, scores, _ = self.predictor.predict(**predict_kwargs)

                best_mask = masks[np.argmax(scores)]
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
