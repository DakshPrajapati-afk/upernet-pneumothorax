#!/bin/bash

# Training script for UPerNet on SIIM-ACR Pneumothorax Segmentation
# Data location: /scratch/dpraja12/data/pneumothorax_stage1/siim-acr-pneumothorax/

# Data paths
DATA_DIR="/scratch/dpraja12/data/pneumothorax_stage1/siim-acr-pneumothorax"
TRAIN_CSV="${DATA_DIR}/stage_1_train_images.csv"
IMAGE_DIR="${DATA_DIR}/png_images"
MASK_DIR="${DATA_DIR}/png_masks"

# Output directory (using scratch for more space)
OUTPUT_DIR="/scratch/dpraja12/upernet_outputs_swin_base"

# Training parameters
NUM_GPUS=2
EPOCHS=100
BATCH_SIZE=8  # Per GPU batch size (reduced for Swin Base - uses more memory)
LEARNING_RATE=1e-4
NUM_WORKERS=8

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
    --loss_type combined \
    --val_split 0.2 
    # --resume /scratch/dpraja12/upernet_outputs_swin_base/best_model.pth 

echo "Training completed!"
