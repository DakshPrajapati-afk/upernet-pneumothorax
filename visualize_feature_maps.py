"""
Extract and visualize feature maps from UPerNet internal layers
Shows what the model learns at different stages of the network
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from sklearn.model_selection import train_test_split


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

    return model


class FeatureExtractor:
    """Hook-based feature extractor for intermediate layers"""

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []

    def register_hooks(self, layer_names):
        """Register forward hooks for specified layers"""

        def get_activation(name):
            def hook(module, input, output):
                # Store the output activation
                if isinstance(output, tuple):
                    self.features[name] = output[0].detach()
                elif isinstance(output, list):
                    # For lists, concatenate or take first element
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        self.features[name] = output[0].detach()
                elif isinstance(output, torch.Tensor):
                    self.features[name] = output.detach()
            return hook

        # Register hooks
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract(self, x):
        """Forward pass and extract features"""
        self.features = {}
        with torch.no_grad():
            _ = self.model(pixel_values=x)
        return self.features


def visualize_feature_maps(model, image_processor, image_path, mask_path,
                          output_dir, device='cuda', num_features_per_layer=8):
    """
    Extract and visualize feature maps from different layers of UPerNet
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Load mask for reference
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert('L')
        mask_np = (np.array(mask) > 0).astype(np.uint8)
    else:
        mask_np = None

    # Prepare input
    inputs = image_processor(image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    print(f'Input shape: {pixel_values.shape}')

    # Define layers to visualize (covering different stages of UPerNet)
    layer_names = [
        'backbone.embeddings',  # Initial patch embeddings
        'backbone.encoder.layers.0',  # Stage 1 (low-level features)
        'backbone.encoder.layers.1',  # Stage 2
        'backbone.encoder.layers.2',  # Stage 3
        'backbone.encoder.layers.3',  # Stage 4 (high-level features)
        'decode_head.psp_modules',  # Pyramid Pooling Module
        'decode_head.bottleneck',  # Decoder bottleneck
    ]

    # Create feature extractor
    extractor = FeatureExtractor(model)
    extractor.register_hooks(layer_names)

    # Extract features
    print('\nExtracting features from layers...')
    features = extractor.extract(pixel_values)

    print(f'\nExtracted features from {len(features)} layers:')
    for name, feat in features.items():
        print(f'  {name}: {feat.shape}')

    # Remove hooks
    extractor.remove_hooks()

    # Visualize features
    print('\nGenerating visualizations...')

    # 1. Create overview with original image and selected feature maps
    selected_layers = list(features.keys())[:7]  # Select up to 7 layers

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Chest X-ray', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Ground truth if available
    if mask_np is not None:
        overlay = image_np.copy()
        overlay_img = np.zeros((*mask_np.shape, 4))
        overlay_img[mask_np == 1] = [1, 0, 0, 0.5]
        axes[1].imshow(image_np, cmap='gray')
        axes[1].imshow(overlay_img)
        axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[1].axis('off')
    else:
        axes[1].axis('off')

    # Feature maps from different layers
    for idx, layer_name in enumerate(selected_layers[:7]):
        ax_idx = idx + 2
        if ax_idx >= len(axes):
            break

        feat = features[layer_name]

        # Average across channels or take first channel
        if len(feat.shape) == 4:  # (B, C, H, W)
            feat_map = feat[0].mean(dim=0).cpu().numpy()  # Average across channels
        elif len(feat.shape) == 3:  # (B, N, C) - transformer output
            # Reshape to 2D for visualization
            B, N, C = feat.shape
            H = W = int(np.sqrt(N))
            if H * W == N:
                feat_map = feat[0, :, :].mean(dim=1).reshape(H, W).cpu().numpy()
            else:
                # Take mean across sequence
                feat_map = feat[0].mean(dim=0).cpu().numpy()
                feat_map = np.tile(feat_map.reshape(-1, 1), (1, int(np.sqrt(len(feat_map))))).T
        else:
            continue

        # Normalize
        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)

        # Plot
        im = axes[ax_idx].imshow(feat_map, cmap='viridis')
        layer_short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
        axes[ax_idx].set_title(f'{layer_short_name}\n{feat.shape}', fontsize=11, fontweight='bold')
        axes[ax_idx].axis('off')
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)

    # Hide remaining axes
    for idx in range(len(selected_layers) + 2, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('UPerNet Feature Maps Visualization', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    overview_path = os.path.join(output_dir, 'feature_maps_overview.png')
    plt.savefig(overview_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'Overview saved: {overview_path}')

    # 2. Create detailed visualization for each layer
    for layer_name, feat in features.items():
        layer_safe_name = layer_name.replace('.', '_').replace('/', '_')

        if len(feat.shape) == 4:  # (B, C, H, W)
            B, C, H, W = feat.shape

            # Select subset of channels to visualize
            num_channels = min(num_features_per_layer, C)
            channel_indices = np.linspace(0, C-1, num_channels, dtype=int)

            # Create grid
            cols = 4
            rows = int(np.ceil(num_channels / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)

            for idx, ch_idx in enumerate(channel_indices):
                row = idx // cols
                col = idx % cols

                feat_map = feat[0, ch_idx].cpu().numpy()

                # Normalize
                feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)

                im = axes[row, col].imshow(feat_map, cmap='viridis')
                axes[row, col].set_title(f'Channel {ch_idx}', fontsize=12, fontweight='bold')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046)

            # Hide unused subplots
            for idx in range(num_channels, rows * cols):
                row = idx // cols
                col = idx % cols
                axes[row, col].axis('off')

            plt.suptitle(f'Layer: {layer_name}\nShape: {feat.shape}',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            layer_path = os.path.join(output_dir, f'{layer_safe_name}_channels.png')
            plt.savefig(layer_path, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f'  Saved: {layer_safe_name}_channels.png')

        elif len(feat.shape) == 3:  # (B, N, C) - transformer output
            B, N, C = feat.shape

            # Try to reshape to 2D
            H = W = int(np.sqrt(N))
            if H * W == N:
                # Visualize first few channels reshaped to 2D
                num_channels = min(num_features_per_layer, C)
                channel_indices = np.linspace(0, C-1, num_channels, dtype=int)

                cols = 4
                rows = int(np.ceil(num_channels / cols))

                fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
                if rows == 1:
                    axes = axes.reshape(1, -1)

                for idx, ch_idx in enumerate(channel_indices):
                    row = idx // cols
                    col = idx % cols

                    feat_map = feat[0, :, ch_idx].reshape(H, W).cpu().numpy()

                    # Normalize
                    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)

                    im = axes[row, col].imshow(feat_map, cmap='viridis')
                    axes[row, col].set_title(f'Feature {ch_idx}', fontsize=12, fontweight='bold')
                    axes[row, col].axis('off')
                    plt.colorbar(im, ax=axes[row, col], fraction=0.046)

                # Hide unused subplots
                for idx in range(num_channels, rows * cols):
                    row = idx // cols
                    col = idx % cols
                    axes[row, col].axis('off')

                plt.suptitle(f'Layer: {layer_name}\nShape: {feat.shape}',
                            fontsize=14, fontweight='bold')
                plt.tight_layout()

                layer_path = os.path.join(output_dir, f'{layer_safe_name}_features.png')
                plt.savefig(layer_path, dpi=120, bbox_inches='tight', facecolor='white')
                plt.close()

                print(f'  Saved: {layer_safe_name}_features.png')

    print(f'\nAll visualizations saved to: {output_dir}/')


def main():
    parser = argparse.ArgumentParser(description='Visualize UPerNet feature maps')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--mask_path', type=str, default=None,
                       help='Path to ground truth mask (optional)')
    parser.add_argument('--output_dir', type=str, default='feature_maps',
                       help='Directory to save feature map visualizations')
    parser.add_argument('--num_features', type=int, default=8,
                       help='Number of feature maps to show per layer')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load image processor
    print('Loading image processor...')
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-swin-base")

    # Load model
    model = load_model(args.checkpoint, device)

    # Visualize feature maps
    visualize_feature_maps(
        model,
        image_processor,
        args.image_path,
        args.mask_path,
        args.output_dir,
        device,
        args.num_features
    )

    print('\n✓ Feature map visualization complete!')


if __name__ == '__main__':
    main()
