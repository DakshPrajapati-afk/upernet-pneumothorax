"""
Visualize UPerNet predictions on validation images
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from sklearn.model_selection import train_test_split


def calculate_dice_score(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = pred.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


def calculate_iou(pred, target, smooth=1e-6):
    """Calculate Intersection over Union"""
    pred = pred.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f'Loading checkpoint from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load config and create model
    config = checkpoint['config']
    print(f'Model architecture:')
    print(f'  Hidden size: {config.hidden_size}')
    print(f'  Num labels: {config.num_labels}')

    model = UperNetForSemanticSegmentation(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f'Model loaded from epoch {checkpoint["epoch"]}')
    print(f'Best Val Dice: {checkpoint.get("val_dice", "N/A"):.4f}')
    print(f'Best Val IoU: {checkpoint.get("val_iou", "N/A"):.4f}')

    return model


def visualize_predictions(model, image_processor, val_df, image_dir, mask_dir,
                         output_dir, num_samples=10, device='cuda'):
    """Generate visualizations comparing predictions with ground truth"""

    os.makedirs(output_dir, exist_ok=True)

    # Select diverse samples: some positive, some negative
    positive_samples = val_df[val_df['has_pneumo'] == 1]
    negative_samples = val_df[val_df['has_pneumo'] == 0]

    n_pos = min(num_samples // 2, len(positive_samples))
    n_neg = min(num_samples // 2, len(negative_samples))

    pos_selected = positive_samples.sample(n=n_pos, random_state=42) if n_pos > 0 else pd.DataFrame()
    neg_selected = negative_samples.sample(n=n_neg, random_state=42) if n_neg > 0 else pd.DataFrame()

    samples = pd.concat([pos_selected, neg_selected]).reset_index(drop=True)

    print(f'\nGenerating visualizations for {len(samples)} validation images...')
    print(f'  - {len(pos_selected)} positive (with pneumothorax)')
    print(f'  - {len(neg_selected)} negative (no pneumothorax)')

    all_dice_scores = []
    all_iou_scores = []

    for idx, row in samples.iterrows():
        filename = row['new_filename']
        has_pneumo = row['has_pneumo']

        # Load image
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        # Load ground truth mask
        mask_path = os.path.join(mask_dir, filename)
        mask = Image.open(mask_path).convert('L')
        mask_np = (np.array(mask) > 0).astype(np.uint8)

        # Prepare input
        inputs = image_processor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)

        # Predict
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            # Resize logits to match original mask size
            logits_resized = torch.nn.functional.interpolate(
                logits,
                size=mask_np.shape,
                mode='bilinear',
                align_corners=False
            )
            pred_mask = torch.argmax(logits_resized, dim=1).squeeze(0).cpu().numpy()

        # Calculate metrics
        pred_tensor = torch.from_numpy(pred_mask).float()
        mask_tensor = torch.from_numpy(mask_np).float()
        dice = calculate_dice_score(pred_tensor, mask_tensor)
        iou = calculate_iou(pred_tensor, mask_tensor)

        all_dice_scores.append(dice)
        all_iou_scores.append(iou)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        # Original image
        axes[0, 0].imshow(image_np, cmap='gray')
        axes[0, 0].set_title('Original Chest X-ray', fontsize=16, fontweight='bold', pad=20)
        axes[0, 0].axis('off')

        # Ground truth overlay
        axes[0, 1].imshow(image_np, cmap='gray', alpha=1.0)
        if mask_np.sum() > 0:
            # Create colored overlay
            overlay = np.zeros((*mask_np.shape, 4))
            overlay[mask_np == 1] = [1, 0, 0, 0.5]  # Red with 50% transparency
            axes[0, 1].imshow(overlay)
        axes[0, 1].set_title(f'Ground Truth\n{"Pneumothorax Present" if has_pneumo else "No Pneumothorax"}',
                            fontsize=16, fontweight='bold', pad=20,
                            color='darkred' if has_pneumo else 'darkgreen')
        axes[0, 1].axis('off')

        # Predicted overlay
        axes[1, 0].imshow(image_np, cmap='gray', alpha=1.0)
        if pred_mask.sum() > 0:
            # Create colored overlay
            overlay = np.zeros((*pred_mask.shape, 4))
            overlay[pred_mask == 1] = [0, 0, 1, 0.5]  # Blue with 50% transparency
            axes[1, 0].imshow(overlay)
        axes[1, 0].set_title(f'Model Prediction\nDice: {dice:.4f} | IoU: {iou:.4f}',
                            fontsize=16, fontweight='bold', pad=20)
        axes[1, 0].axis('off')

        # Comparison overlay
        axes[1, 1].imshow(image_np, cmap='gray', alpha=1.0)

        # Create comparison overlay: Green = correct, Red = false positive, Yellow = false negative
        comparison = np.zeros((*pred_mask.shape, 4))

        # True positives (both predict and GT have pneumothorax) - Green
        true_positive = (pred_mask == 1) & (mask_np == 1)
        comparison[true_positive] = [0, 1, 0, 0.6]

        # False positives (predict pneumothorax, but GT is background) - Red
        false_positive = (pred_mask == 1) & (mask_np == 0)
        comparison[false_positive] = [1, 0, 0, 0.6]

        # False negatives (predict background, but GT is pneumothorax) - Yellow
        false_negative = (pred_mask == 0) & (mask_np == 1)
        comparison[false_negative] = [1, 1, 0, 0.6]

        if comparison.sum() > 0:
            axes[1, 1].imshow(comparison)

        # Create legend
        green_patch = mpatches.Patch(color='green', label='True Positive (Correct)')
        red_patch = mpatches.Patch(color='red', label='False Positive (Over-prediction)')
        yellow_patch = mpatches.Patch(color='yellow', label='False Negative (Missed)')
        axes[1, 1].legend(handles=[green_patch, red_patch, yellow_patch],
                         loc='upper right', fontsize=12, framealpha=0.9)

        axes[1, 1].set_title('Comparison Overlay\n(Green=Correct, Red=FP, Yellow=FN)',
                            fontsize=16, fontweight='bold', pad=20)
        axes[1, 1].axis('off')

        # Add overall title
        status = "POSITIVE CASE" if has_pneumo else "NEGATIVE CASE"
        quality = "Excellent" if dice > 0.8 else "Good" if dice > 0.6 else "Fair" if dice > 0.4 else "Poor"
        fig.suptitle(f'{status} - {filename}\nPrediction Quality: {quality}',
                    fontsize=18, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Save figure
        output_path = os.path.join(output_dir, f'prediction_{idx+1:02d}_{filename}')
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f'  [{idx+1}/{len(samples)}] Saved: {os.path.basename(output_path)} (Dice: {dice:.4f}, IoU: {iou:.4f})')

    # Print summary statistics
    print(f'\n{"="*70}')
    print(f'SUMMARY STATISTICS ({len(samples)} validation samples)')
    print(f'{"="*70}')
    print(f'Average Dice Score: {np.mean(all_dice_scores):.4f} ± {np.std(all_dice_scores):.4f}')
    print(f'Average IoU Score:  {np.mean(all_iou_scores):.4f} ± {np.std(all_iou_scores):.4f}')
    print(f'Best Dice:  {np.max(all_dice_scores):.4f}')
    print(f'Worst Dice: {np.min(all_dice_scores):.4f}')
    print(f'{"="*70}')

    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Dice scores histogram
    axes[0].hist(all_dice_scores, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(all_dice_scores), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(all_dice_scores):.4f}')
    axes[0].set_xlabel('Dice Score', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=14, fontweight='bold')
    axes[0].set_title('Distribution of Dice Scores', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].grid(alpha=0.3)

    # IoU scores histogram
    axes[1].hist(all_iou_scores, bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(all_iou_scores), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(all_iou_scores):.4f}')
    axes[1].set_xlabel('IoU Score', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=14, fontweight='bold')
    axes[1].set_title('Distribution of IoU Scores', fontsize=16, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'summary_statistics.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'\nSummary plot saved: {summary_path}')
    print(f'All visualizations saved to: {output_dir}/')


def main():
    parser = argparse.ArgumentParser(description='Visualize UPerNet predictions')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--train_csv', type=str,
                       default='siim-acr-pneumothorax-data/stage_1_train_images.csv',
                       help='Path to training CSV')
    parser.add_argument('--image_dir', type=str,
                       default='siim-acr-pneumothorax-data/png_images',
                       help='Directory containing images')
    parser.add_argument('--mask_dir', type=str,
                       default='siim-acr-pneumothorax-data/png_masks',
                       help='Directory containing masks')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (must match training)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (must match training)')
    parser.add_argument('--negative_keep_ratio', type=float, default=1.0,
                       help='Negative keep ratio (must match training)')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data (same preprocessing as training)
    print('Loading data...')
    df = pd.read_csv(args.train_csv)

    # Apply negative undersampling if specified (must match training)
    if args.negative_keep_ratio < 1.0:
        print(f'Applying negative undersampling (keep ratio: {args.negative_keep_ratio})...')
        positives = df[df['has_pneumo'] == 1]
        negatives = df[df['has_pneumo'] == 0]
        keep_n_negatives = int(len(negatives) * args.negative_keep_ratio)
        negatives_sampled = negatives.sample(n=keep_n_negatives, random_state=args.seed)
        df = pd.concat([positives, negatives_sampled]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Split (must match training split)
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=args.seed)
    print(f'Validation set: {len(val_df)} images')
    print(f'  Positive: {val_df["has_pneumo"].sum()}')
    print(f'  Negative: {len(val_df) - val_df["has_pneumo"].sum()}')

    # Load image processor
    print('Loading image processor...')
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-swin-base")

    # Load model
    model = load_model(args.checkpoint, device)

    # Generate visualizations
    visualize_predictions(
        model,
        image_processor,
        val_df,
        args.image_dir,
        args.mask_dir,
        args.output_dir,
        args.num_samples,
        device
    )

    print('\n✓ Visualization complete!')


if __name__ == '__main__':
    main()
