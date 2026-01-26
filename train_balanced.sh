#!/bin/bash

# Training script for UPerNet on SIIM-ACR Pneumothorax Segmentation
# WITH CLASS IMBALANCE HANDLING (WeightedRandomSampler + Class Weights + Focal-Dice Loss)
# Data location: /scratch/dpraja12/data/pneumothorax_stage1/siim-acr-pneumothorax/

# Data paths
DATA_DIR="/scratch/dpraja12/data/pneumothorax_stage1/siim-acr-pneumothorax"
TRAIN_CSV="${DATA_DIR}/stage_1_train_images.csv"
IMAGE_DIR="${DATA_DIR}/png_images"
MASK_DIR="${DATA_DIR}/png_masks"

# Output directory (using scratch for more space)
OUTPUT_DIR="/scratch/dpraja12/upernet_outputs_balanced"

# Training parameters
NUM_GPUS=2
EPOCHS=100
BATCH_SIZE=4  # Per GPU batch size (reduced for Swin Base - uses more memory)
LEARNING_RATE=1e-4
NUM_WORKERS=8

# Class imbalance handling parameters
POSITIVE_WEIGHT=10.0       # Sample positive cases 10x more frequently (severe imbalance)
FOCAL_ALPHA=0.75          # Weight for positive class in Focal Loss (higher for severe imbalance)
FOCAL_GAMMA=2.5           # Focusing parameter (higher = more focus on hard examples)
CLASS_WEIGHT_SAMPLES=1000 # Number of samples to use for calculating class weights

echo "======================================"
echo "Starting UPerNet Training with Class Imbalance Handling"
echo "======================================"
echo "Configuration:"
echo "  - WeightedRandomSampler: ENABLED (positive_weight=${POSITIVE_WEIGHT})"
echo "  - Class Weights: ENABLED (${CLASS_WEIGHT_SAMPLES} samples)"
echo "  - Loss Function: Focal-Dice Loss (alpha=${FOCAL_ALPHA}, gamma=${FOCAL_GAMMA})"
echo "  - Aggressive Augmentation: ENABLED"
echo "  - Pretrained Backbone: ENABLED"
echo "  - GPUs: ${NUM_GPUS}"
echo "  - Batch Size: ${BATCH_SIZE} per GPU"
echo "  - Epochs: ${EPOCHS}"
echo "======================================"

# Run training
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
    --pretrained_backbone \
    --val_split 0.2 \
    --use_weighted_sampler \
    --positive_weight ${POSITIVE_WEIGHT} \
    --use_class_weights \
    --class_weight_samples ${CLASS_WEIGHT_SAMPLES} \
    --loss_type focal_dice \
    --focal_alpha ${FOCAL_ALPHA} \
    --focal_gamma ${FOCAL_GAMMA} \
    --aggressive_aug 
    # --flexible_resume \
    # --resume "${OUTPUT_DIR}/best_model.pth" 
    # --use_wandb \
    # --wandb_project pneumothorax-segmentation \
    # --experiment_name upernet-swin-balanced
    

echo ""
echo "======================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "======================================"
