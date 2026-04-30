import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
from tqdm import tqdm

class Dice(nn.Module):
    """
    Dice coefficient calculator for binary segmentation tasks.
    """
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, predict, target):
        smooth = 1
        intersection = (predict * target).sum()
        dice = (2. * intersection + smooth) / (predict.sum() + target.sum() + smooth)
        return dice

class HD95(nn.Module):
    """
    HD95 calculator for binary segmentation tasks.
    使用距离变换方法计算Hausdorff距离
    """
    def __init__(self):
        super(HD95, self).__init__()

    def forward(self, predict, target):
        return self.calculate_hd(predict, target)

    def calculate_hd(self, predict, target):
        # 检查是否存在有效的分割区域
        if predict.sum() == 0:
            # 如果预测为空，设置一个点
            predict = predict.clone()
            predict[0, 0, 0] = 1.0
        if target.sum() == 0:
            # 如果目标为空，设置一个点
            target = target.clone()
            target[0, 0, 0] = 1.0
        
        # 转换为numpy数组用于计算
        pred_np = predict.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # 使用距离变换计算HD距离
        right_hd = self.hd_distance(pred_np, target_np)
        left_hd = self.hd_distance(target_np, pred_np)
        
        # 返回最大距离
        hausdorff_distance = max(right_hd, left_hd)
        return torch.tensor(hausdorff_distance, dtype=torch.float32)
    
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))
        return np.array(np.percentile(distances[indexes], 95))

def evaluate_model(net, dataloader, device):
    """
    Evaluate the model on both Dice coefficient and HD95 metric in a single inference pass.
    Returns: (dice_score, hd95_score, dice_values, hd95_values)
    """
    net.eval()
    dice_calculator = Dice()
    hd_calculator = HD95()

    dice_values = []
    hd95_values = []

    for batch in dataloader:
        try:
            if isinstance(batch, dict):
                image = batch['image']
                mask_true = batch['label']
            else:
                image = batch[0]
                mask_true = batch[1]
            image = image.to(device=device)
            mask_true = mask_true.to(device=device)
            batch_size = image.size(0)

            with torch.no_grad():
                try:
                    mask_pred = net(image, batch=batch)
                except TypeError:
                    mask_pred = net(image)
                if isinstance(mask_pred, list):
                    mask_pred = mask_pred[0]
                mask_pred = F.sigmoid(mask_pred)
                mask_pred_binary = (mask_pred > 0.5).float()

                for i in range(batch_size):
                    try:
                        pred_mask = mask_pred_binary[i]
                        true_mask = (mask_true[i] > 0.5).float()
                        dice_value = dice_calculator(pred_mask, true_mask).item()
                        hd95_value = hd_calculator(pred_mask, true_mask).item()
                        dice_values.append(dice_value)
                        hd95_values.append(hd95_value)
                    except Exception as e:
                        print(f"Error calculating metrics for sample {i}: {e}")
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue

    if len(dice_values) == 0:
        print("Warning: No valid Dice values calculated!")
        dice_score = 0.0
    else:
        dice_score = round(float(np.mean(dice_values)), 8)

    if len(hd95_values) == 0:
        print("Warning: No valid HD values calculated!")
        hd95_score = 0.0
    else:
        hd95_score = round(float(np.mean(hd95_values)), 4)

    return dice_score, hd95_score, dice_values, hd95_values
