#!/bin/bash

# Training script for UPerNet on SIIM-ACR Pneumothorax Segmentation
# WITH NEGATIVE UNDERSAMPLING (removes most negative samples)

# Data paths
DATA_DIR="/scratch/dpraja12/data/pneumothorax_stage1/siim-acr-pneumothorax"
TRAIN_CSV="${DATA_DIR}/stage_1_train_images.csv"
IMAGE_DIR="${DATA_DIR}/png_images"
MASK_DIR="${DATA_DIR}/png_masks"

# Output directory
OUTPUT_DIR="/scratch/dpraja12/upernet_outputs_balanced_v2_resume"

# Training parameters
NUM_GPUS=3
EPOCHS=200
BATCH_SIZE=4  # Per GPU batch size (reduced from 5 to save memory)
LEARNING_RATE=1e-4
NUM_WORKERS=4  # Reduced from 8 to save RAM

# BALANCED CLASS IMBALANCE HANDLING (gentler approach)
POSITIVE_WEIGHT=1.5       # Much gentler oversampling
FOCAL_ALPHA=0.5           # Moderate positive class weight
FOCAL_GAMMA=2.0           # Standard focal parameter
CLASS_WEIGHT_SAMPLES=1000

# NEGATIVE UNDERSAMPLING RATIO
# Keep only 1 negative for every 2 positives
# If you have 2500 positives, keep ~1250 negatives
# This dramatically reduces the 8000:2500 ratio to 1250:2500 (1:2)

echo "======================================"
echo "Resuming UPerNet Training with Balanced Sampling v2"
echo "======================================"
echo "Configuration:"
echo "  - Strategy: Negative Undersampling (0.15) + Gentle Oversampling"
echo "  - Negative Keep Ratio: 0.15 (keeps only 15% of negatives)"
echo "  - Positive Weight: ${POSITIVE_WEIGHT} (gentle)"
echo "  - Class Weights: ENABLED"
echo "  - Loss: Combined Dice+CE"
echo "  - Augmentation: STANDARD (not aggressive)"
echo "  - GPUs: ${NUM_GPUS}"
echo "  - Batch Size: ${BATCH_SIZE} per GPU"
echo "  - Expected Dataset: ~2500 pos + ~1200 neg = ~3700 total"
echo "  - Resuming from: ${OUTPUT_DIR}/best_model.pth"
echo "======================================"

# Run training with resume
python3 train_upernet.py \
    --train_csv "${TRAIN_CSV}" \
    --image_dir "${IMAGE_DIR}" \
    --mask_dir "${MASK_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_gpus ${NUM_GPUS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_workers ${NUM_WORKERS} \
    --val_split 0.2 \
    --use_class_weights \
    --class_weight_samples ${CLASS_WEIGHT_SAMPLES} \
    --loss_type combined \
    --negative_keep_ratio 0.15 \
    --resume "/scratch/dpraja12/upernet_outputs_balanced_v2/best_model.pth"

echo ""
echo "======================================"
echo "Training completed!"
echo "======================================"
