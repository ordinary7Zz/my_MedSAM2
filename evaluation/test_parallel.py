import argparse
import os
import sys
import torch
import numpy as np
import time
import logging
import imageio
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from metrics import evaluate_model
from dataset import FullDataset
from medsam2_seg_wrapper import MedSAM2SegWrapper
from medsam2_seg_wrapper_refine import MedSAM2SegWrapperRefine

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def clean_path(path):
    """Clean path by removing extra quotes and whitespace."""
    if isinstance(path, str):
        # Remove quotes if present
        if (path.startswith('"') and path.endswith('"')) or \
           (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        # Strip whitespace
        path = path.strip()
    return path

def mean_ci_95(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    margin = 1.96 * std / np.sqrt(values.size)
    return mean, mean - margin, mean + margin


def process_dataset(model, image_path, gt_path, save_base_path, dataset_name, device, save_results):
    # Create save directory for this dataset
    save_path = os.path.join(save_base_path, dataset_name)
    if save_results.lower() == "true":
        os.makedirs(save_path, exist_ok=True)
    
    # Additional path cleaning
    image_path = clean_path(image_path)
    gt_path = clean_path(gt_path)
    
    print(f"Processing dataset: {dataset_name}")
    print(f"Image path: {image_path}")
    print(f"GT path: {gt_path}")
    print(f"Save path: {save_path}")
    
    # Check path existence before loading dataset
    if not os.path.exists(image_path):
        print(f"Error: Image directory does not exist: {image_path}")
        return 0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    if not os.path.exists(gt_path):
        print(f"Error: Mask directory does not exist: {gt_path}")
        return 0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    
    # List directory contents for debugging
    try:
        image_files = os.listdir(image_path)
        mask_files = os.listdir(gt_path)
        print(f"Found {len(image_files)} files in image directory")
        print(f"Found {len(mask_files)} files in mask directory")
    except Exception as e:
        print(f"Error listing directory contents: {e}")
    
    model = model.to(device)
    
    # 加载测试数据集，使用与训练时相同的尺寸
    target_size = 224  
    print(f"Loading dataset with FullDataset using size: {target_size}x{target_size}")
    test_dataset = FullDataset(image_path, gt_path, target_size, mode='val')
    
    # 检查数据集是否为空
    if len(test_dataset) == 0:
        print(f"Error: Test dataset {dataset_name} is empty!")
        return 0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    print(f"Dataset loaded: {len(test_dataset)} images found")
    
    # 计算评估指标
    print(f"Calculating evaluation metrics for dataset: {dataset_name}")
    
    # 计算Dice分数和95%置信区间
    dice_score, hd95_score, dice_values, hd95_values = evaluate_model(model, test_loader, device)
    dice_mean, dice_ci_low, dice_ci_high = mean_ci_95(dice_values)
    hd95_mean, hd95_ci_low, hd95_ci_high = mean_ci_95(hd95_values)
    print(f"Dice Score: {dice_score:.8f}, 95% CI: [{dice_ci_low:.4f}, {dice_ci_high:.4f}]")
    print(f"HD95: {hd95_score}, 95% CI: [{hd95_ci_low:.4f}, {hd95_ci_high:.4f}]")
    
    # 保存预测结果（如果需要）
    if save_results.lower() == "true":
        model.eval()
        for i, batch in enumerate(tqdm(test_loader, desc='Saving predictions', unit='image')):
            with torch.no_grad():
                image = batch['image'].to(device=device)
                name = batch.get('name', [f'image_{i}'])[0]
                
                # Forward pass
                try:
                    res = model(image, batch=batch)
                except TypeError:
                    res = model(image)
                if type(res) == type([]):
                    res = res[0]
                
                # 后处理和保存
                res_sigmoid = res.sigmoid().data.cpu()
                res_np = res_sigmoid.numpy().squeeze()
                res_normalized = (res_np - res_np.min()) / (res_np.max() - res_np.min() + 1e-8)
                res_uint8 = (res_normalized * 255).astype(np.uint8)
                
                # 使用从数据集中获取的原始文件名（不包含扩展名）
                output_filename = f"{name}.png"
                try:
                    imageio.imsave(os.path.join(save_path, output_filename), res_uint8)
                    # print(f"Saved prediction: {output_filename}")
                except Exception as e:
                    print(f"Error saving prediction for {name}: {e}")
    print(f"Dataset {dataset_name} processing completed.")
    
    return dice_score, hd95_score, (dice_mean, dice_ci_low, dice_ci_high), (hd95_mean, hd95_ci_low, hd95_ci_high)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser("MedSAM2 Segmentation Test")
    parser.add_argument("--checkpoint", "--checkpoint_path", type=str, required=True,
                    help="path to the MedSAM2 checkpoint (.pt)")
    parser.add_argument("--sam2_cfg", type=str, default="sam2.1_hiera_t512.yaml",
                        help="SAM2/MedSAM2 config name (e.g. sam2.1_hiera_t512.yaml)")
    parser.add_argument('--test_image_paths', '--image_paths', type=str, action='append', default=[],
                        help='paths to the test image directories (can be used multiple times)')
    parser.add_argument('--test_gt_paths', '--gt_paths', '--test_mask_paths', type=str, action='append', default=[],
                        help='paths to the test mask directories (can be used multiple times)')
    parser.add_argument('--test_dataset_names', '--dataset_names', type=str, action='append', default=[],
                        help='names of the test datasets (can be used multiple times)')
    parser.add_argument("--save_path", type=str, default="./predictions",
                        help="base path to save the predicted masks")
    parser.add_argument("--save_results", type=str, default="true",
                        help="Whether to save prediction results (true/false)")
    parser.add_argument("--cuda_visible_devices", type=str, default=None,
                        help="CUDA visible devices setting")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory to save log files")
    args = parser.parse_args()
    
    # Handle CUDA device settings
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    
    # Configure logging system
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"test_{timestamp}.log")
    
    # Set up logger
    sys.stdout = Logger(log_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    
    # Log configuration
    print(f"Checkpoint path: {args.checkpoint}")
    print(f"Save path: {args.save_path}")
    print(f"Save results: {args.save_results}")
    print(f"Test dataset names: {args.test_dataset_names}")
    
    # Ensure test dataset paths are provided
    if not args.test_image_paths or not args.test_gt_paths:
        print("Error: No test datasets provided. Please use --test_image_paths and --test_gt_paths arguments.")
        return
    
    # Clean test paths by removing any extra quotes
    args.test_image_paths = [clean_path(path) for path in args.test_image_paths]
    args.test_gt_paths = [clean_path(path) for path in args.test_gt_paths]
    
    # Log information about test paths
    print(f"Number of test image paths: {len(args.test_image_paths) if args.test_image_paths else 0}")
    if args.test_image_paths:
        print(f"Test image paths: {args.test_image_paths}")
        # Validate each path and log existence
        for i, path in enumerate(args.test_image_paths):
            if os.path.exists(path):
                print(f"Image path {i+1} exists: {path}")
            else:
                print(f"Warning: Image path {i+1} does not exist: {path}")
    
    print(f"Number of test ground truth paths: {len(args.test_gt_paths) if args.test_gt_paths else 0}")
    if args.test_gt_paths:
        print(f"Test ground truth paths: {args.test_gt_paths}")
        # Validate each path and log existence
        for i, path in enumerate(args.test_gt_paths):
            if os.path.exists(path):
                print(f"GT path {i+1} exists: {path}")
            else:
                print(f"Warning: GT path {i+1} does not exist: {path}")
    
    # Ensure test image paths and mask paths数量匹配
    if len(args.test_image_paths) != len(args.test_gt_paths):
        print(f"Warning: Number of test image paths ({len(args.test_image_paths)}) and mask paths ({len(args.test_gt_paths)}) do not match.")
        print("Using the minimum number of pairs.")
        min_len = min(len(args.test_image_paths), len(args.test_gt_paths))
        args.test_image_paths = args.test_image_paths[:min_len]
        args.test_gt_paths = args.test_gt_paths[:min_len]
    
    # Create base save directory
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Created base save directory: {args.save_path}")

    # Build MedSAM2 model once
    print("Loading MedSAM2 model...")
    try:
        model = MedSAM2SegWrapper(
            sam2_cfg=args.sam2_cfg,
            checkpoint_path=clean_path(args.checkpoint),
            device=device
        )
        print("MedSAM2 model loaded.")
        print("Prompt mode: using GT mask -> box prompt for MedSAM2.")
    except Exception as e:
        print(f"Error initializing MedSAM2: {e}")
        return
    
    # Storage for all datasets' metrics
    all_metrics = []
    
    # Process each test dataset
    start_time = time.time()
    for i, (img_path, gt_path) in enumerate(zip(args.test_image_paths, args.test_gt_paths)):
        # Use provided dataset name if available, otherwise use default naming
        if i < len(args.test_dataset_names) and args.test_dataset_names[i]:
            dataset_name = args.test_dataset_names[i]
        else:
            dataset_name = f"Test_Set_{i+1}"
        
        print(f"\nProcessing dataset {i+1}/{len(args.test_image_paths)}")
        dice, hd95, dice_ci, hd95_ci = process_dataset(model, img_path, gt_path, args.save_path, dataset_name, device, args.save_results)
        all_metrics.append((dataset_name, dice, hd95, dice_ci, hd95_ci))
    
    total_time = time.time() - start_time
    print(f"All datasets processed in {total_time:.2f} seconds")
    
    # Print summary of all datasets
    print("\n===== Summary of All Datasets =====")
    for dataset_name, dice, hd95, dice_ci, hd95_ci in all_metrics:
        print(f"Dataset: {dataset_name}")
        print(f"  Dice Score: {dice:.8f}")
        print(f"  Dice 95% CI: [{dice_ci[1]:.4f}, {dice_ci[2]:.4f}]")
        print(f"  HD95: {hd95}")
        print(f"  HD95 95% CI: [{hd95_ci[1]:.4f}, {hd95_ci[2]:.4f}]")
    
    print("\nAll datasets processing completed successfully!")
    print(f"Log file location: {log_file}")
    
    # Close logger properly
    try:
        if hasattr(sys.stdout, 'close') and hasattr(sys.stdout, '_closed') and not sys.stdout._closed:
            sys.stdout.close()
    except Exception as e:
        # Just handle the exception to avoid script failure
        pass

if __name__ == "__main__":
    main()