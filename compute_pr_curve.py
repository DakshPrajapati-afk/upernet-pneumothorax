"""
Compute Precision-Recall Curve for UPerNet Pneumothorax Segmentation
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from tqdm import tqdm


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


def compute_pr_curve(model, image_processor, val_df, image_dir, mask_dir,
                     output_dir, device='cuda', max_samples=None):
    """
    Compute Precision-Recall curve by collecting predictions across all validation samples
    """

    os.makedirs(output_dir, exist_ok=True)

    print(f'\nComputing PR curve on validation set...')
    print(f'Total validation images: {len(val_df)}')

    if max_samples is not None:
        val_df = val_df.sample(n=min(max_samples, len(val_df)), random_state=42)
        print(f'Using {len(val_df)} samples for PR curve computation')

    # Collect all predictions and ground truth
    all_probs = []
    all_targets = []

    positive_count = 0
    negative_count = 0

    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc='Processing images'):
        filename = row['new_filename']
        has_pneumo = row['has_pneumo']

        # Load image
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert('RGB')

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

            # Get probability for pneumothorax class (class 1)
            probs = torch.softmax(logits_resized, dim=1)
            pneumo_probs = probs[0, 1].cpu().numpy()  # Shape: (H, W)

        # Flatten and collect
        all_probs.append(pneumo_probs.flatten())
        all_targets.append(mask_np.flatten())

        if has_pneumo:
            positive_count += 1
        else:
            negative_count += 1

    print(f'\nProcessed {len(val_df)} images:')
    print(f'  - Positive cases: {positive_count}')
    print(f'  - Negative cases: {negative_count}')

    # Concatenate all predictions and targets
    print('\nConcatenating predictions...')
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    print(f'Total pixels: {len(all_probs):,}')
    print(f'Positive pixels (pneumothorax): {all_targets.sum():,} ({100*all_targets.sum()/len(all_targets):.2f}%)')
    print(f'Negative pixels (background): {(1-all_targets).sum():,} ({100*(1-all_targets).sum()/len(all_targets):.2f}%)')

    # Compute Precision-Recall curve
    print('\nComputing Precision-Recall curve...')
    precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)

    # Compute Average Precision
    ap_score = average_precision_score(all_targets, all_probs)
    pr_auc = auc(recall, precision)

    print(f'\nMetrics:')
    print(f'  Average Precision (AP): {ap_score:.4f}')
    print(f'  PR-AUC: {pr_auc:.4f}')

    # Find optimal threshold (maximum F1-score)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    print(f'\nOptimal Operating Point (max F1):')
    print(f'  Threshold: {optimal_threshold:.4f}')
    print(f'  Precision: {optimal_precision:.4f}')
    print(f'  Recall: {optimal_recall:.4f}')
    print(f'  F1-Score: {optimal_f1:.4f}')

    # Save PR curve data
    pr_data = {
        'precision': precision[:-1].tolist(),
        'recall': recall[:-1].tolist(),
        'thresholds': thresholds.tolist(),
        'ap_score': ap_score,
        'pr_auc': pr_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_precision': optimal_precision,
        'optimal_recall': optimal_recall,
        'optimal_f1': optimal_f1,
    }

    np.save(os.path.join(output_dir, 'pr_curve_data.npy'), pr_data)
    print(f'\nPR curve data saved to: {os.path.join(output_dir, "pr_curve_data.npy")}')

    # Plot Precision-Recall curve
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PR Curve
    axes[0].plot(recall, precision, linewidth=2, label=f'PR Curve (AP={ap_score:.4f}, AUC={pr_auc:.4f})')
    axes[0].scatter([optimal_recall], [optimal_precision], color='red', s=100, zorder=5,
                   label=f'Optimal (F1={optimal_f1:.4f}, Thr={optimal_threshold:.3f})')
    axes[0].set_xlabel('Recall', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Precision', fontsize=14, fontweight='bold')
    axes[0].set_title('Precision-Recall Curve\nPneumothorax Segmentation', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12, loc='lower left')
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    # F1-Score vs Threshold
    axes[1].plot(thresholds, f1_scores, linewidth=2, color='green', label='F1-Score')
    axes[1].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal Threshold: {optimal_threshold:.3f}')
    axes[1].set_xlabel('Threshold', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    axes[1].set_title('F1-Score vs Decision Threshold', fontsize=16, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    pr_curve_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(pr_curve_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'PR curve plot saved to: {pr_curve_path}')

    # Create detailed metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Precision vs Threshold
    axes[0, 0].plot(thresholds, precision[:-1], linewidth=2, color='blue')
    axes[0, 0].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Threshold', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Precision vs Threshold', fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([0, 1])

    # Recall vs Threshold
    axes[0, 1].plot(thresholds, recall[:-1], linewidth=2, color='orange')
    axes[0, 1].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Threshold', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Recall', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Recall vs Threshold', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 1])

    # F1-Score vs Threshold (zoomed)
    axes[1, 0].plot(thresholds, f1_scores, linewidth=2, color='green')
    axes[1, 0].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
    axes[1, 0].axhline(optimal_f1, color='red', linestyle=':', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Threshold', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'F1-Score vs Threshold (Max F1={optimal_f1:.4f})', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].set_ylim([0, 1])

    # Precision-Recall tradeoff
    axes[1, 1].plot(recall[:-1], precision[:-1], linewidth=2, color='purple')
    axes[1, 1].scatter([optimal_recall], [optimal_precision], color='red', s=150, zorder=5)
    axes[1, 1].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'PR Curve (AP={ap_score:.4f})', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])

    plt.tight_layout()
    detailed_path = os.path.join(output_dir, 'detailed_metrics.png')
    plt.savefig(detailed_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'Detailed metrics plot saved to: {detailed_path}')

    # Save summary to text file
    summary_path = os.path.join(output_dir, 'pr_curve_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('='*70 + '\n')
        f.write('PRECISION-RECALL CURVE ANALYSIS\n')
        f.write('Pneumothorax Segmentation Model\n')
        f.write('='*70 + '\n\n')

        f.write(f'Dataset Statistics:\n')
        f.write(f'  Total validation images: {len(val_df)}\n')
        f.write(f'  Positive cases: {positive_count}\n')
        f.write(f'  Negative cases: {negative_count}\n')
        f.write(f'  Total pixels analyzed: {len(all_probs):,}\n')
        f.write(f'  Pneumothorax pixels: {all_targets.sum():,} ({100*all_targets.sum()/len(all_targets):.2f}%)\n')
        f.write(f'  Background pixels: {(1-all_targets).sum():,} ({100*(1-all_targets).sum()/len(all_targets):.2f}%)\n\n')

        f.write(f'Model Performance:\n')
        f.write(f'  Average Precision (AP): {ap_score:.4f}\n')
        f.write(f'  PR-AUC: {pr_auc:.4f}\n\n')

        f.write(f'Optimal Operating Point (Maximum F1-Score):\n')
        f.write(f'  Decision Threshold: {optimal_threshold:.4f}\n')
        f.write(f'  Precision: {optimal_precision:.4f}\n')
        f.write(f'  Recall: {optimal_recall:.4f}\n')
        f.write(f'  F1-Score: {optimal_f1:.4f}\n\n')

        f.write('='*70 + '\n')

    print(f'Summary saved to: {summary_path}')

    return pr_data


def main():
    parser = argparse.ArgumentParser(description='Compute PR curve for UPerNet')

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
    parser.add_argument('--output_dir', type=str, default='pr_curve_results',
                       help='Directory to save PR curve results')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (must match training)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (must match training)')
    parser.add_argument('--negative_keep_ratio', type=float, default=1.0,
                       help='Negative keep ratio (must match training)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (default: all)')

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

    # Compute PR curve
    pr_data = compute_pr_curve(
        model,
        image_processor,
        val_df,
        args.image_dir,
        args.mask_dir,
        args.output_dir,
        device,
        args.max_samples
    )

    print('\n✓ PR curve computation complete!')


if __name__ == '__main__':
    main()
