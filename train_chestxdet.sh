#!/bin/bash

# Training script for UPerNet on ChestX-Det10 Multi-Disease Segmentation
# Data location: /scratch/dpraja12/data/chestxdet/

# Data paths
DATA_DIR="/scratch/dpraja12/data/chestxdet"
TRAIN_JSON="${DATA_DIR}/ChestX-Det10-Dataset/train.json"
IMAGE_DIR="${DATA_DIR}/train/train-old"

# Output directory
OUTPUT_DIR="/scratch/dpraja12/upernet_outputs_chestxdet"

# Training parameters
NUM_GPUS=1
EPOCHS=100
BATCH_SIZE=8  # Per GPU batch size (multi-class is more memory intensive)
LEARNING_RATE=1e-4
NUM_WORKERS=8

# Class imbalance handling parameters
POSITIVE_WEIGHT=8.0        # Sample images with annotations 8x more frequently
FOCAL_ALPHA=0.25          # Weight for disease classes in Focal Loss
FOCAL_GAMMA=2.5           # Focusing parameter
CLASS_WEIGHT_SAMPLES=500  # Number of samples to use for calculating class weights

echo "======================================"
echo "Starting UPerNet Training for ChestX-Det10"
echo "======================================"
echo "Configuration:"
echo "  - Dataset: ChestX-Det10 (13 diseases)"
echo "  - Model: UPerNet with Swin-Base backbone"
echo "  - Classes: 14 (13 diseases + background)"
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
python3 train_upernet_chestxdet.py \
    --train_json "${TRAIN_JSON}" \
    --image_dir "${IMAGE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_gpus ${NUM_GPUS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_workers ${NUM_WORKERS} \
    --pretrained_backbone \
    --val_split 0.2 \
    --use_class_weights \
    --class_weight_samples ${CLASS_WEIGHT_SAMPLES} \
    --loss_type focal_dice \
    --focal_alpha ${FOCAL_ALPHA} \
    --focal_gamma ${FOCAL_GAMMA} \
    --aggressive_aug
    # --use_weighted_sampler \
    # --positive_weight ${POSITIVE_WEIGHT} \
    # --use_wandb \
    # --wandb_project chestxdet-segmentation \
    # --experiment_name upernet-swin-chestxdet
    # --resume "${OUTPUT_DIR}/best_model.pth"

echo ""
echo "======================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "======================================"
