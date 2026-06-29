#!/bin/bash

# ---------------------- Configuration ----------------------
# Set CUDA device
CUDA_VISIBLE_DEVICES="1"

# Model checkpoint path
CHECKPOINT_PATH="/mnt/wangbd8/workspace/ThyroidAgent/MedSAM2/my_finetune/MedSAM2_TG_Video/checkpoints/checkpoint_10.pt"

# Configure multiple test dataset paths
# 测试数据集名称数组
TEST_DATASET_NAMES=(
    "TGVideo"
)

# 测试图像路径数组
TEST_IMAGE_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/image/"
)

# 测试掩码路径数组
TEST_MASK_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TGVideo_PNG/test/mask/"
)

# Ensure arrays have the same length
if [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_IMAGE_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_MASK_PATHS[@]} ]; then
    echo "Error: Arrays must have the same length"
    exit 1
fi

# Prediction results save path
SAVE_PATH="./predictions/finetune/TGVideo"

# Whether to save prediction results (true/false)
SAVE_RESULTS="false"

# Log directory
LOG_DIR="./logs/test_logs/TGVideo"

# ---------------------- Execution ----------------------
# Set CUDA environment variable
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

# Ensure repo root is on PYTHONPATH (so evaluation scripts can import sam2/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

# Create save directory if it doesn't exist
mkdir -p "$SAVE_PATH"

# Build test image paths arguments
TEST_IMAGE_ARGS=()
for img_path in "${TEST_IMAGE_PATHS[@]}"; do
    if [ -d "$img_path" ]; then
        TEST_IMAGE_ARGS+=("--test_image_paths" "$img_path")
    fi
done

# Build test mask paths arguments
TEST_MASK_ARGS=()
for mask_path in "${TEST_MASK_PATHS[@]}"; do
    if [ -d "$mask_path" ]; then
        TEST_MASK_ARGS+=("--test_gt_paths" "$mask_path")
    fi
done

# Build test dataset names arguments
TEST_NAMES_ARGS=()
for dataset_name in "${TEST_DATASET_NAMES[@]}"; do
    TEST_NAMES_ARGS+=("--test_dataset_names" "$dataset_name")
done

# Execute the test command
CMD="python -u \"$SCRIPT_DIR/test_parallel.py\" \
    --checkpoint \"$CHECKPOINT_PATH\" \
    --sam2_cfg \"configs/sam2.1_hiera_t512.yaml\" \
    ${TEST_IMAGE_ARGS[@]} \
    ${TEST_MASK_ARGS[@]} \
    ${TEST_NAMES_ARGS[@]} \
    --save_path \"$SAVE_PATH\" \
    --save_results \"$SAVE_RESULTS\" \
    --log_dir \"$LOG_DIR\""

echo "Executing command: $CMD"
eval $CMD
