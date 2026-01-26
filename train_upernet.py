"""
Training script for UPerNet on SIIM-ACR Pneumothorax Segmentation (Stage 1)
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation
from transformers import AutoImageProcessor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb


def patch_adaptive_avg_pool2d_for_mps():
    """
    Monkey-patch adaptive_avg_pool2d to handle MPS incompatibility.
    When input sizes are not divisible by output sizes on MPS, fall back to CPU.
    """
    import torch.nn.functional as F
    original_adaptive_avg_pool2d = F.adaptive_avg_pool2d

    def mps_safe_adaptive_avg_pool2d(input, output_size):
        if input.device.type == 'mps':
            # Move to CPU, apply pooling, move back to MPS wha
            input_cpu = input.to('cpu')
            output_cpu = original_adaptive_avg_pool2d(input_cpu, output_size)
            return output_cpu.to('mps')
        else:
            return original_adaptive_avg_pool2d(input, output_size)

    F.adaptive_avg_pool2d = mps_safe_adaptive_avg_pool2d
    torch.nn.functional.adaptive_avg_pool2d = mps_safe_adaptive_avg_pool2d


class PneumothoraxDataset(Dataset):
    """Dataset for Pneumothorax segmentation from PNG images and masks (Stage 1)"""

    def __init__(self, df, image_dir, mask_dir, image_processor, transform=None, is_train=True):
        """
        Args:
            df: DataFrame with new_filename, ImageId, and has_pneumo columns
            image_dir: Directory containing PNG images
            mask_dir: Directory containing PNG masks
            image_processor: HuggingFace image processor for model input
            transform: Optional augmentation transforms
            is_train: Whether this is training data
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_processor = image_processor
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['new_filename']

        # Load PNG image
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Load PNG mask
        mask_path = os.path.join(self.mask_dir, filename)
        mask = Image.open(mask_path).convert('L')  # Grayscale
        mask = np.array(mask)

        # Convert mask to binary (0 or 1)
        # Assuming mask is already binary, but ensure it's 0 or 1
        mask = (mask > 0).astype(np.uint8)

        # Apply augmentations if provided
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Process image for model input
        # Note: image_processor expects PIL Image or numpy array
        encoded_inputs = self.image_processor(image, return_tensors="pt")

        # Remove batch dimension added by processor
        pixel_values = encoded_inputs['pixel_values'].squeeze(0)

        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask).long()

        return {
            'pixel_values': pixel_values,
            'labels': mask_tensor,
            'image_id': row['ImageId'],
            'filename': filename
        }


def get_augmentation_pipeline(aggressive=False):
    """
    Create augmentation pipeline using albumentations

    Args:
        aggressive: If True, use more aggressive augmentations for imbalanced datasets
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if aggressive:
            # More aggressive augmentations to create more variety from positive samples
            train_transform = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=20,
                    p=0.7,
                    border_mode=0
                ),
                A.OneOf([
                    A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(p=1),
                    A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1),
                ], p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                    A.GaussianBlur(blur_limit=(3, 7), p=1),
                    A.MotionBlur(blur_limit=5, p=1),
                ], p=0.4),
                A.OneOf([
                    A.CLAHE(clip_limit=4.0, p=1),
                    A.Sharpen(p=1),
                    A.Emboss(p=1),
                ], p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.3
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=3,
                    min_height=8,
                    min_width=8,
                    p=0.3
                ),
            ])
        else:
            # Standard augmentations
            train_transform = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.OneOf([
                    A.GaussNoise(p=1),
                    A.GaussianBlur(p=1),
                ], p=0.3),
            ])

        val_transform = A.Compose([
            A.Resize(512, 512),
        ])

        return train_transform, val_transform
    except ImportError:
        print("Warning: albumentations not installed. Install with: pip install albumentations")
        print("Using basic transforms only")
        return None, None


def calculate_dice_score(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


def calculate_iou(pred, target, smooth=1e-6):
    """Calculate Intersection over Union"""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def create_weighted_sampler(df, oversample_positive=True, positive_weight=3.0):
    """
    Create a WeightedRandomSampler to balance positive and negative samples.

    Args:
        df: DataFrame with 'has_pneumo' column
        oversample_positive: If True, oversample positive cases
        positive_weight: How much more to sample positive cases (e.g., 3.0 = 3x more)

    Returns:
        WeightedRandomSampler
    """
    # Calculate sample weights based on has_pneumo
    weights = []
    for idx in range(len(df)):
        has_pneumo = df.iloc[idx]['has_pneumo']
        if has_pneumo:
            weights.append(positive_weight)  # Higher weight for positive samples
        else:
            weights.append(1.0)  # Normal weight for negative samples

    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    return sampler


def load_checkpoint_flexible(model, checkpoint_path, device, strict=False):
    """
    Load checkpoint with flexible weight loading.
    Only loads weights that match in size, ignores mismatched layers.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        strict: If True, requires exact match (default: False)

    Returns:
        dict: Checkpoint metadata (epoch, metrics, etc.)
    """
    print(f'Loading checkpoint from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if strict:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Load weights flexibly - only compatible layers
        model_state = model.state_dict()
        checkpoint_state = checkpoint['model_state_dict']

        loaded_keys = []
        skipped_keys = []

        for key, param in checkpoint_state.items():
            if key in model_state:
                if model_state[key].shape == param.shape:
                    model_state[key] = param
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch: {param.shape} vs {model_state[key].shape})")
            else:
                skipped_keys.append(f"{key} (not in model)")

        model.load_state_dict(model_state)

        print(f'Successfully loaded {len(loaded_keys)}/{len(checkpoint_state)} layers')
        if skipped_keys:
            print(f'Skipped {len(skipped_keys)} incompatible layers:')
            for key in skipped_keys[:5]:  # Show first 5
                print(f'  - {key}')
            if len(skipped_keys) > 5:
                print(f'  ... and {len(skipped_keys) - 5} more')

    return checkpoint


def calculate_class_weights(df, image_dir, mask_dir, num_samples=1000):
    """
    Calculate class weights based on pixel frequency in the dataset.

    Args:
        df: DataFrame with image information
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        num_samples: Number of samples to use for estimation (None = use all)

    Returns:
        torch.Tensor: Class weights [background_weight, pneumothorax_weight]
    """
    print(f"Calculating class weights from {num_samples if num_samples else 'all'} samples...")

    total_background = 0
    total_pneumo = 0

    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42) if num_samples else df

    for idx in tqdm(range(len(sample_df)), desc="Sampling masks"):
        row = sample_df.iloc[idx]
        filename = row['new_filename']
        mask_path = os.path.join(mask_dir, filename)

        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)

        total_background += (mask == 0).sum()
        total_pneumo += (mask == 1).sum()

    # Calculate weights inversely proportional to class frequency
    total_pixels = total_background + total_pneumo

    if total_pneumo == 0:
        print("Warning: No positive pixels found in sampled data!")
        return torch.tensor([1.0, 1.0])

    weight_background = total_pixels / (2 * total_background)
    weight_pneumo = total_pixels / (2 * total_pneumo)

    class_weights = torch.tensor([weight_background, weight_pneumo], dtype=torch.float32)

    print(f"Total background pixels: {total_background:,}")
    print(f"Total pneumothorax pixels: {total_pneumo:,}")
    print(f"Class imbalance ratio: {total_background / total_pneumo:.2f}:1")
    print(f"Calculated class weights: background={weight_background:.4f}, pneumothorax={weight_pneumo:.4f}")

    return class_weights


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)[:, 1]  # Get probability of class 1
        target = target.float()

        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, H, W) labels
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined Cross Entropy and Dice Loss"""

    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()

    def forward(self, logits, labels):
        ce = self.ce_loss(logits, labels)
        dice = self.dice_loss(logits, labels)
        return self.ce_weight * ce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    """Combined Focal Loss and Dice Loss for severe class imbalance"""

    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2.0):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss()

    def forward(self, logits, labels):
        focal = self.focal_loss(logits, labels)
        dice = self.dice_loss(logits, labels)
        return self.focal_weight * focal + self.dice_weight * dice


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, rank=0, is_distributed=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    # Only show progress bar on main process
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    else:
        pbar = dataloader

    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        # Skip batches with size 1 due to BatchNorm requirements
        if pixel_values.size(0) == 1:
            print("\nWarning: Skipping batch with size 1 (BatchNorm requires batch_size > 1)")
            continue

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)

        # Calculate loss
        loss = criterion(outputs.logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            pred_masks = torch.argmax(outputs.logits, dim=1)
            dice = calculate_dice_score(pred_masks, labels)
            iou = calculate_iou(pred_masks, labels)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'iou': f'{iou:.4f}'
            })

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_dice, avg_iou


def validate(model, dataloader, criterion, device, epoch, rank=0, is_distributed=False):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    # Only show progress bar on main process
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Validation')
    else:
        pbar = dataloader

    with torch.no_grad():
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)

            # Calculate loss
            loss = criterion(outputs.logits, labels)

            # Calculate metrics
            pred_masks = torch.argmax(outputs.logits, dim=1)
            dice = calculate_dice_score(pred_masks, labels)
            iou = calculate_iou(pred_masks, labels)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou

            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice:.4f}',
                    'iou': f'{iou:.4f}'
                })

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_dice, avg_iou


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def main_worker(rank, world_size, args):
    """Main worker function for each GPU process"""
    is_distributed = world_size > 1

    # Setup distributed training if using multiple GPUs
    if is_distributed:
        setup_distributed(rank, world_size)

    # Only print on main process
    if rank == 0:
        print(f'Using {world_size} GPU(s) for training')

    # Validate batch size (must be >= 2 due to BatchNorm)
    if args.batch_size < 2:
        if rank == 0:
            print(f'Error: batch_size must be >= 2 (got {args.batch_size})')
            print('This is required because UPerNet uses Batch Normalization layers.')
            print('Use --batch_size 2 or higher.')
        if is_distributed:
            cleanup_distributed()
        return

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device(f'cuda:{rank}')
    if rank == 0:
        print(f'Device: {device}')

    # Initialize wandb if enabled (only on main process)
    if args.use_wandb and rank == 0:
        wandb.init(project=args.wandb_project, name=args.experiment_name, config=vars(args))

    # Load data
    if rank == 0:
        print('Loading data...')
    df = pd.read_csv(args.train_csv)

    # Apply negative undersampling if specified
    if args.negative_keep_ratio < 1.0:
        if rank == 0:
            print(f'\nApplying negative undersampling (keep ratio: {args.negative_keep_ratio})...')

        # Separate positive and negative samples
        positives = df[df['has_pneumo'] == 1]
        negatives = df[df['has_pneumo'] == 0]

        if rank == 0:
            print(f'Original dataset: {len(positives)} positives, {len(negatives)} negatives')
            print(f'Imbalance ratio: {len(negatives) / len(positives):.2f}:1 (neg:pos)')

        # Sample subset of negatives
        keep_n_negatives = int(len(negatives) * args.negative_keep_ratio)
        negatives_sampled = negatives.sample(n=keep_n_negatives, random_state=args.seed)

        # Recombine and shuffle
        df = pd.concat([positives, negatives_sampled]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

        if rank == 0:
            print(f'Filtered dataset: {len(positives)} positives, {len(negatives_sampled)} negatives')
            print(f'New imbalance ratio: {len(negatives_sampled) / len(positives):.2f}:1 (neg:pos)')
            print(f'Total samples after filtering: {len(df)} (removed {len(negatives) - len(negatives_sampled)} negatives)\n')

    # Split into train and validation
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=args.seed)

    if rank == 0:
        print(f'Train images: {len(train_df)}, Validation images: {len(val_df)}')

    # Create image processor
    if rank == 0:
        print('Setting up image processor...')
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-swin-base")

    # Get augmentation pipeline
    train_transform, val_transform = get_augmentation_pipeline(aggressive=args.aggressive_aug)

    # Create datasets
    if rank == 0:
        print('Creating datasets...')
    train_dataset = PneumothoraxDataset(
        train_df,
        args.image_dir,
        args.mask_dir,
        image_processor,
        transform=train_transform,
        is_train=True
    )

    val_dataset = PneumothoraxDataset(
        val_df,
        args.image_dir,
        args.mask_dir,
        image_processor,
        transform=val_transform,
        is_train=False
    )

    # Create dataloaders with distributed sampler if using multiple GPUs
    # Note: WeightedRandomSampler cannot be used with DistributedSampler
    if args.use_weighted_sampler and not is_distributed:
        if rank == 0:
            print(f'Using WeightedRandomSampler with positive weight: {args.positive_weight}')
        train_sampler = create_weighted_sampler(train_df, positive_weight=args.positive_weight)
        shuffle_train = False
    elif is_distributed:
        if rank == 0 and args.use_weighted_sampler:
            print('Warning: WeightedRandomSampler not compatible with DistributedSampler. Using DistributedSampler instead.')
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        shuffle_train = False
    else:
        train_sampler = None
        shuffle_train = True

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    if rank == 0:
        print('Initializing model...')

    # If resuming, load config from checkpoint first (only on rank 0 to save memory)
    if args.resume:
        if rank == 0:
            print(f'Loading config from checkpoint: {args.resume}')
            checkpoint_for_config = torch.load(args.resume, map_location='cpu', weights_only=False)
            config = checkpoint_for_config['config']
            print(f'Creating model with checkpoint architecture...')
            print(f'  Hidden size: {config.hidden_size}')
            print(f'  Num labels: {config.num_labels}')

            # Extract essential config info for broadcasting (avoid pickling large objects)
            config_dict = {
                'hidden_size': config.hidden_size,
                'num_labels': config.num_labels,
                'backbone_config': config.backbone_config.to_dict() if hasattr(config, 'backbone_config') else None,
            }
            del checkpoint_for_config  # Free memory
        else:
            config_dict = None

        # Broadcast config dict from rank 0 to all other ranks
        if is_distributed:
            if rank == 0:
                config_list = [config_dict]
            else:
                config_list = [None]
            dist.broadcast_object_list(config_list, src=0)
            config_dict = config_list[0]

        # Reconstruct config from dict on each rank
        if rank != 0:
            # Non-rank-0 processes need to reconstruct the config
            from transformers import SwinConfig
            if config_dict['backbone_config']:
                backbone_config = SwinConfig.from_dict(config_dict['backbone_config'])
            else:
                # Fallback - create a config based on hidden size
                if config_dict['hidden_size'] == 512:
                    backbone_config = SwinConfig.from_pretrained(
                        "microsoft/swin-tiny-patch4-window7-224",
                        out_features=["stage1", "stage2", "stage3", "stage4"]
                    )
                else:
                    backbone_config = SwinConfig.from_pretrained(
                        "microsoft/swin-base-patch4-window12-384",
                        out_features=["stage1", "stage2", "stage3", "stage4"]
                    )

            config = UperNetConfig(
                backbone_config=backbone_config,
                num_labels=config_dict['num_labels'],
                hidden_size=config_dict['hidden_size'],
            )

        model = UperNetForSemanticSegmentation(config)
    # Load pretrained backbone if specified (only if not resuming)
    elif args.pretrained_backbone:
        if rank == 0:
            print('Loading pretrained UperNet with Swin-Base backbone...')
        # Load the full pretrained model
        pretrained = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-base")

        # Modify the decoder head for our binary segmentation task
        pretrained.config.num_labels = 2

        # Get the correct input channels from the existing classifier layers
        decode_in_channels = pretrained.decode_head.classifier.in_channels
        aux_in_channels = pretrained.auxiliary_head.classifier.in_channels

        # Replace classifiers with binary segmentation heads
        pretrained.decode_head.classifier = nn.Conv2d(
            decode_in_channels,
            2,  # Binary segmentation
            kernel_size=1
        )
        pretrained.auxiliary_head.classifier = nn.Conv2d(
            aux_in_channels,
            2,  # Binary segmentation
            kernel_size=1
        )

        model = pretrained
        config = pretrained.config
    else:
        # Create from scratch with window size 12 to match pretrained weights if needed later
        backbone_config = SwinConfig.from_pretrained(
            "microsoft/swin-base-patch4-window12-384",
            out_features=["stage1", "stage2", "stage3", "stage4"]
        )

        config = UperNetConfig(
            backbone_config=backbone_config,
            num_labels=2,  # Binary segmentation: background and pneumothorax
            hidden_size=1024,  # Swin-Base has hidden_size of 1024 (vs 768 for Tiny)
        )

        model = UperNetForSemanticSegmentation(config)

    model = model.to(device)

    # Wrap model with DDP if using multiple GPUs
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Calculate class weights if enabled
    class_weights = None
    if args.use_class_weights:
        if rank == 0:
            print('Calculating class weights...')
        class_weights = calculate_class_weights(
            train_df,
            args.image_dir,
            args.mask_dir,
            num_samples=args.class_weight_samples
        )
        class_weights = class_weights.to(device)
        if rank == 0:
            print(f'Class weights: {class_weights}')

    # Setup loss and optimizer
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss_type == 'dice':
        criterion = DiceLoss()
    elif args.loss_type == 'combined':
        criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=class_weights)
    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss_type == 'focal_dice':
        criterion = FocalDiceLoss(
            focal_weight=0.5,
            dice_weight=0.5,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    best_dice = 0.0

    if args.resume:
        if rank == 0:
            print(f'Resuming from checkpoint: {args.resume}')

        try:
            # Try strict loading first
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

            # Load model state dict (handle DDP vs non-DDP models)
            if is_distributed:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer only if architectures match (strict loading succeeded)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('val_dice', 0.0)

            if rank == 0:
                print(f'Successfully resumed from epoch {checkpoint["epoch"]}')
                print(f'Previous best Dice: {best_dice:.4f}')
                print(f'Previous Val IoU: {checkpoint.get("val_iou", "N/A"):.4f}')

        except RuntimeError as e:
            if "size mismatch" in str(e) and args.flexible_resume:
                # Architecture mismatch - try flexible loading
                if rank == 0:
                    print('\nArchitecture mismatch detected!')
                    print('Using flexible weight loading (only compatible layers)...')
                    print('Note: Training will start from epoch 1 with reset optimizer.\n')

                # Use flexible loading function
                target_model = model.module if is_distributed else model
                checkpoint = load_checkpoint_flexible(target_model, args.resume, device, strict=False)

                # Don't restore optimizer or epoch (different architecture)
                start_epoch = 1
                best_dice = 0.0

                if rank == 0:
                    print(f'Loaded backbone weights from checkpoint (epoch {checkpoint["epoch"]})')
                    print('Starting fresh training with new architecture.')
            else:
                # Re-raise error if not flexible resume or different error
                if rank == 0:
                    print(f'\nError loading checkpoint: {e}')
                    print('Tip: Use --flexible_resume to load only compatible weights from different architecture.')
                raise

    # Learning rate scheduler (initialized after resume to set correct last_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
        last_epoch=start_epoch - 2  # -2 because scheduler starts at -1, and we want it at start_epoch - 1
    )

    # Training loop
    if rank == 0:
        print('Starting training...')

    for epoch in range(start_epoch, args.epochs + 1):
        if rank == 0:
            print(f'\nEpoch {epoch}/{args.epochs}')

        # Set epoch for distributed sampler (ensures different shuffle each epoch)
        if is_distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, rank, is_distributed
        )

        # Validate
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device, epoch, rank, is_distributed
        )

        # Update learning rate
        scheduler.step()

        # Print metrics (only on main process)
        if rank == 0:
            print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Log to wandb (only on main process)
        if args.use_wandb and rank == 0:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou,
                'lr': optimizer.param_groups[0]['lr']
            })

        # Save best model (only on main process)
        if val_dice > best_dice and rank == 0:
            best_dice = val_dice
            print(f'New best model! Dice: {best_dice:.4f}')

            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            temp_save_path = save_path + '.tmp'

            try:
                # Save model state dict (unwrap DDP if necessary)
                model_state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                    'val_iou': val_iou,
                    'config': config,
                }, temp_save_path)
                os.replace(temp_save_path, save_path)
                print(f'Model saved to {save_path}')
            except Exception as e:
                print(f'Warning: Failed to save best model: {e}')
                if os.path.exists(temp_save_path):
                    os.remove(temp_save_path)

        # Save checkpoint (only on main process)
        if epoch % args.save_every == 0 and rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            temp_checkpoint_path = checkpoint_path + '.tmp'

            try:
                # Save to temporary file first (unwrap DDP if necessary)
                model_state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                    'val_iou': val_iou,
                }, temp_checkpoint_path)

                # If successful, rename to final path (atomic operation)
                os.replace(temp_checkpoint_path, checkpoint_path)
                print(f'Checkpoint saved to {checkpoint_path}')

                # Clean up old checkpoints to save space (keep only last 2)
                if args.keep_last_n_checkpoints > 0:
                    all_checkpoints = sorted([
                        f for f in os.listdir(args.output_dir)
                        if f.startswith('checkpoint_epoch_') and f.endswith('.pth')
                    ], key=lambda x: int(x.split('_')[-1].split('.')[0]))

                    if len(all_checkpoints) > args.keep_last_n_checkpoints:
                        for old_checkpoint in all_checkpoints[:-args.keep_last_n_checkpoints]:
                            old_path = os.path.join(args.output_dir, old_checkpoint)
                            os.remove(old_path)
                            print(f'Removed old checkpoint: {old_checkpoint}')

            except Exception as e:
                print(f'Warning: Failed to save checkpoint: {e}')
                if os.path.exists(temp_checkpoint_path):
                    os.remove(temp_checkpoint_path)

    if rank == 0:
        print(f'\nTraining completed! Best Dice: {best_dice:.4f}')

    if args.use_wandb and rank == 0:
        wandb.finish()

    # Cleanup distributed training
    if is_distributed:
        cleanup_distributed()


def main(args):
    """Main entry point that handles single vs multi-GPU training"""
    if args.num_gpus > 1:
        # Multi-GPU training with DDP
        world_size = args.num_gpus
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # Single GPU training
        main_worker(0, 1, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UPerNet for Pneumothorax Segmentation (Stage 1)')

    # Data parameters
    parser.add_argument('--train_csv', type=str, default='siim-acr-pneumothorax-data/stage_1_train_images.csv',
                        help='Path to training CSV file')
    parser.add_argument('--image_dir', type=str, default='siim-acr-pneumothorax-data/png_images',
                        help='Directory containing PNG images')
    parser.add_argument('--mask_dir', type=str, default='siim-acr-pneumothorax-data/png_masks',
                        help='Directory containing PNG masks')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save model checkpoints')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')

    # Model parameters
    parser.add_argument('--pretrained_backbone', action='store_true',
                        help='Use pretrained Swin backbone')
    parser.add_argument('--loss_type', type=str, default='combined',
                        choices=['ce', 'dice', 'combined', 'focal', 'focal_dice'],
                        help='Loss function type')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--flexible_resume', action='store_true',
                        help='Allow resuming from checkpoints with different architectures (loads only compatible weights)')

    # Class imbalance handling parameters
    parser.add_argument('--use_weighted_sampler', action='store_true',
                        help='Use WeightedRandomSampler to oversample positive cases')
    parser.add_argument('--positive_weight', type=float, default=3.0,
                        help='Weight for positive samples in WeightedRandomSampler (default: 3.0)')
    parser.add_argument('--negative_keep_ratio', type=float, default=1.0,
                        help='Fraction of negative samples to keep (default: 1.0 = keep all). Use 0.15 to keep only 15%% of negatives for severe imbalance')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights in loss function based on pixel frequency')
    parser.add_argument('--class_weight_samples', type=int, default=1000,
                        help='Number of samples to use for calculating class weights (default: 1000)')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha parameter for Focal Loss (default: 0.25)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss (default: 2.0)')

    # Data augmentation parameters
    parser.add_argument('--aggressive_aug', action='store_true',
                        help='Use aggressive augmentations for imbalanced datasets')

    # Training parameters
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use for training (use 0 for CPU)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size per GPU (minimum 2 required for BatchNorm)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--keep_last_n_checkpoints', type=int, default=2,
                        help='Keep only last N checkpoints to save disk space (0 to keep all)')

    # Logging parameters
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='pneumothorax-segmentation',
                        help='W&B project name')
    parser.add_argument('--experiment_name', type=str, default='upernet-swin',
                        help='Experiment name')

    args = parser.parse_args()

    main(args)
