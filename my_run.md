# MedSAM2 — 2D 甲状腺超声图像分割 运行指南

本文档说明如何使用 MedSAM2 项目进行 **2D 甲状腺超声图像** 的分割训练与推理。

---

## 目录

- [1. 环境安装](#1-环境安装)
- [2. 数据集准备](#2-数据集准备)
  - [2.1 推理/评估：PNG 格式（推荐）](#21-推理评估png-格式推荐)
  - [2.2 训练：NPZ 格式](#22-训练npz-格式)
- [3. 推理（使用已有模型权重）](#3-推理使用已有模型权重)
  - [3.1 快速批量推理](#31-快速批量推理)
  - [3.2 单张图像推理](#32-单张图像推理)
- [4. 微调（Fine-tuning）](#4-微调fine-tuning)
  - [4.0 微调原理概述](#40-微调原理概述)
  - [4.1 下载预训练权重](#41-下载预训练权重)
  - [4.2 微调配文件（已提供）](#42-微调配文件已提供)
  - [4.3 数据集划分](#43-数据集划分)
  - [4.4 启动微调训练](#44-启动微调训练)
  - [4.5 训练输出](#45-训练输出)
  - [4.6 微调后评估](#46-微调后评估)
  - [4.7 超参数调优建议](#47-超参数调优建议)
- [5. 常见问题](#5-常见问题)

---

## 1. 环境安装

```bash
# 1. 创建 conda 环境
conda create -n medsam2 python=3.12 -y
conda activate medsam2

# 2. 安装 PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 MedSAM2
cd /path/to/MedSAM2
pip install -e ".[dev]"

# 4. 下载模型权重
bash download.sh
```

---

## 2. 数据集准备

你的 2D 甲状腺超声数据集应包含 **图像** 和对应的 **分割标注 (GT mask)**。

### 2.1 推理/评估：PNG 格式（推荐）

适用于直接推理或评估已有模型。将图像和 mask 分别放在两个目录下，**按文件名（不含扩展名）一一对应**。

#### 目录结构

```
your_dataset/
├── images/                     # 超声原图
│   ├── case_001.png
│   ├── case_002.png
│   ├── case_003.jpg            # 支持 jpg/png/bmp/tif/webp
│   └── ...
├── masks/                      # 对应的 GT mask（二值图）
│   ├── case_001.png
│   ├── case_002.png
│   ├── case_003.png            # 文件名（不含扩展名）必须与图像一致
│   └── ...
```

#### 要求

| 项目 | 要求 |
|------|------|
| **图像格式** | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp` |
| **Mask 格式** | `.png` 等，灰度图 (L mode) |
| **Mask 内容** | 二值图：前景（甲状腺结节）= 255（或 > 128），背景 = 0 |
| **文件名** | 图像与 mask 的 stem 必须一致，如 `case_001.png` ↔ `case_001.png` |
| **图像通道** | 超声为灰度图，代码会自动转为 RGB（三通道复制） |

---

### 2.2 训练：NPZ 格式

训练使用 `.npz` 格式，每个 `.npz` 包含一个 2D 图像（`D=1`，即单帧"视频"），配置中 `num_frames=1`。

#### .npz 文件结构

```python
data = np.load('thyroid_001.npz')
print(data.keys())   # ['imgs', 'gts']
```

| 键名 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `imgs` | `(1, H, W)` | `uint8` | 灰度超声图像，值域 [0, 255] |
| `gts`  | `(1, H, W)` | `uint8/int` | 整数 mask，0=背景，1=病灶 |

#### 使用 `png_to_npz.py`

```bash
python png_to_npz.py \
    --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/image \
    --mask_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/mask  \
    --output_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/train/npz/npz_MedSAM2
```

脚本自动按文件名 stem 匹配 image-mask pair，每对生成一个独立的 `.npz` 文件。

---

## 3. 推理（使用已有模型权重）

### 3.1 快速批量推理

使用 `evaluation/test_parallel.py` 对一个或多个测试集进行推理并计算 Dice / HD95 指标。

```bash
# 确保项目根目录在 PYTHONPATH 中
export PYTHONPATH="/path/to/MedSAM2:${PYTHONPATH}"

python evaluation/test_parallel.py \
    --checkpoint /path/to/MedSAM2_latest.pt \
    --sam2_cfg configs/sam2.1_hiera_t512.yaml \
    --test_image_paths /path/to/your_dataset/images/ \
    --test_gt_paths    /path/to/your_dataset/masks/  \
    --test_dataset_names ThyroidTest \
    --save_path ./predictions/ \
    --save_results true \
    --log_dir ./logs/
```

| 参数 | 说明 |
|------|------|
| `--checkpoint` | 训练好的 MedSAM2 模型权重 (.pt) |
| `--sam2_cfg` | 模型配置文件，通常在 `sam2/configs/` 下 |
| `--test_image_paths` | 测试图像目录（可多次使用以添加多个数据集） |
| `--test_gt_paths` | 对应 GT mask 目录 |
| `--test_dataset_names` | 数据集名称（用于日志和输出子目录） |
| `--save_results` | `true` 保存预测图，`false` 仅打印指标 |
| `--log_dir` | 日志输出目录 |

**输出**：
- 控制台打印每个数据集的 Dice Score、HD95 及 95% 置信区间
- 若 `--save_results true`，预测的 sigmoid mask 以 PNG 保存在 `--save_path/<dataset_name>/` 下

### 3.2 批量推理（无 GT mask，全图 Box）

如果你只有图像没有 GT mask，使用 `infer_2d.py`，以**全图作为 box prompt** 进行推理：

```bash
python infer_2d.py \
    --image_dir /path/to/images/ \
    --checkpoint /path/to/medsam2.pt \
    --output_dir ./predictions/
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image_dir` | 必填 | 输入图像目录 |
| `--checkpoint` | 必填 | MedSAM2 权重路径 (.pt) |
| `--output_dir` | `./predictions` | 输出 mask 目录 |
| `--sam2_cfg` | `configs/sam2.1_hiera_t512.yaml` | 模型配置 |
| `--device` | `cuda` | 推理设备 |
| `--ext` | `.png .jpg .jpeg .bmp .tif .tiff` | 图像扩展名 |

输出 mask 自动保持原图尺寸，以二值 PNG 保存在 `--output_dir` 下。

---

## 4. 微调（Fine-tuning）

本节详细介绍如何在 2D 甲状腺超声数据上微调 MedSAM2。

### 4.0 微调原理概述

MedSAM2 的模型架构包含三部分：

| 组件 | 说明 | 微调策略 |
|------|------|----------|
| **Image Encoder** (Hiera) | 提取图像特征 | 继续训练，学习率降低 10 倍（`vision_lr`） |
| **Memory Attention** | 帧间 memory 传播 | 2D 任务无帧间依赖，但仍保留 |
| **SAM Mask Decoder** | 从 prompt + 特征生成 mask | 全力训练（`base_lr`） |

**训练流程**：每张 2D 图像 → NPZRawDataset 读入（D=1 即单帧"视频"） → 从 GT mask 生成 box prompt → SAM2 decoder 预测 mask → 计算 loss（mask + dice + iou）。

### 4.1 下载预训练权重

微调需要两个权重文件：

```bash
mkdir -p checkpoints

# 1. SAM2.1 原始权重（模型架构初始化用）
wget -O checkpoints/sam2.1_hiera_tiny.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# 2. MedSAM2 预训练权重（微调起点，医学图像领域已预训练）
# 运行 download.sh 或从 HuggingFace 下载
bash download.sh
# 下载后 checkpoints/medsam2_latest.pt 即为 MedSAM2 权重
```

> **两个权重的区别**：
> - `sam2.1_hiera_tiny.pt`：Meta 在自然图像上训练的原始 SAM2.1 权重
> - `medsam2_latest.pt`：在 CT/MRI/超声等医学 3D 数据上进一步训练的 MedSAM2 权重，作为微调起点效果更好

### 4.2 微调配文件（已提供）

项目已包含适配 2D 甲状腺超声的配置文件：

```
sam2/configs/sam2.1_hiera_tiny512_thyroid2d.yaml
```

你**只需修改文件最顶部的 2 行**：

```yaml
user:
  data_folder: /PATH/TO/YOUR/NPZ/DIRECTORY    # 改为你的 NPZ 数据集绝对路径
  checkpoint:   /PATH/TO/medsam2_latest.pt     # 改为你的 MedSAM2 权重路径
```

文件中其余位置通过 `${user.data_folder}` 和 `${user.checkpoint}` 自动引用，无需修改。

#### 配置文件关键参数一览

```yaml
scratch:
  resolution: 512                # 所有图像 resize 到 512×512
  train_video_batch_size: 2      # 每 GPU batch size，单帧约 8GB 显存可设到 4
  num_frames: 1                  # ★ 2D 图像 = 1 帧视频
  max_num_objects: 1             # ★ 每张图一个甲状腺结节目标
  base_lr: 5.0e-5                # decoder 学习率
  vision_lr: 3.0e-05             # image encoder 学习率（更低以保护预训练特征）
  num_epochs: 50                 # 2D 微调 30~50 epoch

trainer.model:
  prob_to_use_box_input_for_train: 1.0   # ★ 始终用 box prompt
  prob_to_use_pt_input_for_train: 0.0    # ★ 关闭独立 point prompt
  num_init_cond_frames_for_train: 1      # ★ 只有 1 个条件帧
  rand_init_cond_frames_for_train: False # ★ 固定为第 1 帧
  num_frames_to_correct_for_train: 1     # iterative correction 最多 1 帧
  rand_frames_to_correct_for_train: False
  num_correction_pt_per_frame: 7         # 保留 correction 机制增加鲁棒性
```

### 4.3 数据集划分

```
your_dataset/
├── train_npz/                          # 训练集 NPZ
│   ├── case_001.npz                    # imgs (1, H, W), gts (1, H, W)
│   ├── case_002.npz
│   └── ...
├── test_images/                        # 测试集图像 (PNG, 用于评估)
│   ├── case_101.png
│   └── ...
└── test_masks/                         # 测试集 GT (PNG)
    ├── case_101.png
    └── ...
```

### 4.4 启动微调训练

```bash
export PYTHONPATH="/path/to/MedSAM2:${PYTHONPATH}"

# 单卡训练
CUDA_VISIBLE_DEVICES=0 python training/train.py \
    -c configs/sam2.1_hiera_tiny512_thyroid2d.yaml \
    --output-path ./my_finetune/MedSAM2_TG_Video \
    --use-cluster 0 \
    --num-gpus 1 \
    --num-nodes 1
CUDA_VISIBLE_DEVICES=0 python training/train.py \
    -c configs/sam2.1_hiera_tiny512_noudle.yaml \
    --output-path ./my_finetune/MedSAM2_TG_Video \
    --use-cluster 0 \
    --num-gpus 1 \
    --num-nodes 1
```

### 4.5 训练输出

```
./exp_log/MedSAM2_Thyroid2D/
├── checkpoints/
│   ├── checkpoint_5.pt       # 每 5 epoch 保存一次
│   ├── checkpoint_10.pt
│   └── ...
├── tensorboard/              # TensorBoard 日志
│   └── events.out.tfevents...
├── logs/                     # 文本日志
│   └── log.txt
├── config.yaml               # 原始配置
└── config_resolved.yaml      # 解析后配置
```

**监控训练**：
```bash
tensorboard --logdir ./exp_log/MedSAM2_Thyroid2D/tensorboard
```

### 4.6 微调后评估

用训练产出的 checkpoint 在测试集上评估：

```bash
python evaluation/test_parallel.py \
    --checkpoint ./exp_log/MedSAM2_Thyroid2D/checkpoints/checkpoint_50.pt \
    --sam2_cfg configs/sam2.1_hiera_t512.yaml \
    --test_image_paths /path/to/your_dataset/test_images/ \
    --test_gt_paths    /path/to/your_dataset/test_masks/  \
    --test_dataset_names ThyroidTest \
    --save_path ./predictions/ \
    --save_results true
```

### 4.7 超参数调优建议

| 场景 | 调整项 | 建议值 |
|------|--------|--------|
| **小数据集 (<200 张)** | `num_epochs` | 30 |
| | `base_lr` | 2e-5 |
| | 数据增强 | 增加 `RandomAffine` 的 degrees/shear |
| **大数据集 (>500 张)** | `train_video_batch_size` | 4~8 |
| | `num_epochs` | 50 |
| **显存不足** | `resolution` | 448 |
| | `train_video_batch_size` | 1 |
| **过拟合** | 增强 ColorJitter、RandomAffine 幅度 | |
| | 降低 `base_lr`、`vision_lr` | |
| **欠拟合** | 增加 `num_epochs` | 75 |
| | 提高 `base_lr` | 1e-4 |

---

## 5. 常见问题

### Q1: 2D 图像为什么还需要 video predictor？

MedSAM2 模型是作为 video model 训练的，其 memory attention 机制依赖 video predictor 的完整 pipeline。即使处理单张 2D 图像，也需要将其作为"1 帧视频"来初始化 state 并传入 prompt。

### Q2: 训练时 num_frames 设置为 1 是否可行？

可行。当 `num_frames=1` 时，训练仅使用单帧的 box prompt，不涉及帧间 memory 传播。这对于独立 2D 图像的分割任务是合适的。

### Q3: 显存不足怎么办？

```yaml
# 减小 batch size
scratch:
  train_video_batch_size: 1

# 或降低分辨率
scratch:
  resolution: 448
```

### Q4: 推理时 Box Prompt 从哪里来？

推理需要提供 box（边界框）或 point prompt：
- **自动提取**：如果有 GT mask，可从 mask 计算 bbox（`evaluation/medsam2_seg_wrapper.py` 中的 `_mask_to_box`）。
- **手动标注**：提供 `[x0, y0, x1, y1]` 像素坐标。
- **检测器**：使用目标检测模型（如 YOLO、DINO）先检测结节位置，再将检测框作为 prompt 输入。

### Q5: 如何只保存预测 mask 不计算指标？

修改 `test_parallel.py` 或直接使用 `medsam2_seg_wrapper.py` 中的 `MedSAM2SegWrapper` 类进行单张推理。

### Q6: 模型如何处理不同尺寸的图像？

所有图像在预处理时会被 **resize 到 512×512**，预测结果也会在原始尺寸上输出。如果原始图像不是正方形，可能会产生轻微变形，建议将原始图像预处理为正方形或接近正方形的尺寸。

---

## 数据流水线总结

```
原始超声图像 (灰度 PNG/JPG)
    │
    ├─→ [训练] → python png_to_npz.py → 打包为 .npz
    │           → 修改 YAML 配置文件中的路径
    │           → python training/train.py
    │           → 输出 checkpoint
    │
    ├─→ [推理/评估] → PNG 图像 + PNG mask (分目录)
    │               → python evaluation/test_parallel.py
    │               → 输出 Dice/HD95 指标 + 预测图
    │
    └─→ [单张推理] → 加载 MedSAM2SegWrapper
                    → 提供图像 + box prompt
                    → 输出 mask
```
