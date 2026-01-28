# UPerNet Pneumothorax Segmentation

A deep learning pipeline for pneumothorax detection and segmentation in chest X-ray images using UPerNet with Swin Transformer backbone.

## Overview

This project implements a semantic segmentation model for detecting pneumothorax (collapsed lung) in chest X-ray images from the SIIM-ACR Pneumothorax Segmentation dataset. The model uses **UPerNet (Unified Perceptual Parsing Network)** architecture with a **Swin Transformer Base** backbone from HuggingFace Transformers.

### Dataset Statistics
- **Training Images**: 2,898 (after balancing)
- **Validation Images**: 725
- **Positive Cases**: 2,379 pneumothorax images
- **Negative Cases**: 1,244 (after undersampling from 8,296)
- **Class Imbalance**: 107.45:1 (background to pneumothorax pixels)

## Architecture

- **Model**: UPerNet (HuggingFace `transformers`)
- **Backbone**: Swin Transformer Base (hidden size: 512)
- **Input Size**: 512x512 RGB images
- **Output**: Binary segmentation mask (2 classes: background, pneumothorax)
- **Loss Function**: Combined Dice Loss + Cross-Entropy with class weighting

## Key Features

- **Multi-GPU Training**: Distributed Data Parallel (DDP) support for 2-3 GPUs
- **Class Imbalance Handling**:
  - Negative undersampling (keeps 15% of negatives)
  - Positive sample oversampling (1.5x weight)
  - Class-weighted loss function (54.22:0.50 pneumothorax:background)
- **Data Augmentation**: Albumentations pipeline with rotations, flips, elastic transforms
- **Experiment Tracking**: Weights & Biases (wandb) integration
- **Checkpoint Management**: Automatic best model saving and resumable training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/upernet-pneumothorax.git
cd upernet-pneumothorax

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- PyTorch >= 2.0.0
- CUDA-capable GPU(s) with 16GB+ VRAM recommended
- See `requirements.txt` for full dependencies

## Dataset Setup

1. Download the [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) dataset
2. Convert DICOM images to PNG format (512x512)
3. Organize data as follows:

```
data/
├── siim-acr-png/
│   ├── images/
│   │   ├── train/
│   │   │   ├── image1.png
│   │   │   └── ...
│   │   └── test/
│   └── masks/
│       └── train/
│           ├── image1.png
│           └── ...
```

## Training

### Basic Training
```bash
# Single GPU
python train_upernet.py

# Multi-GPU (recommended)
./train.sh
```

### Balanced Training (Recommended)
```bash
# With class balancing strategies
./train_balanced_v2.sh
```

### Training Configuration

Edit the shell scripts or modify `train_upernet.py` arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100-200 | Number of training epochs |
| `--batch_size` | 4 | Batch size per GPU |
| `--lr` | 1e-4 | Initial learning rate |
| `--image_size` | 512 | Input image resolution |
| `--num_workers` | 4 | Data loading workers |

## Evaluation

### Compute Precision-Recall Curves
```bash
python compute_pr_curve.py --model_path outputs/best_model.pth --output_dir pr_curve_results/
```

### Visualize Predictions
```bash
python visualize_predictions.py --model_path outputs/best_model.pth --output_dir visualizations/
```

### Feature Map Visualization
```bash
python visualize_feature_maps.py --model_path outputs/best_model.pth --output_dir feature_maps/
```

## Project Structure

```
upernet/
├── train_upernet.py          # Main training script
├── train_upernet_chestxdet.py # Multi-class variant for ChestX-Det
├── compute_pr_curve.py       # Evaluation metrics
├── visualize_predictions.py  # Prediction visualization
├── visualize_feature_maps.py # Feature map analysis
├── test_gpu.py               # GPU setup verification
├── train.sh                  # Basic training script
├── train_balanced.sh         # Balanced sampling v1
├── train_balanced_v2.sh      # Balanced sampling v2 (recommended)
├── train_chestxdet.sh        # ChestX-Det training
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Training Strategies

### Handling Class Imbalance

The pneumothorax segmentation task suffers from severe class imbalance (~0.99% pneumothorax pixels vs ~99.01% background). This project addresses it through:

1. **Sample-level balancing**: Undersampling negative images (15% retention) and gentle oversampling of positive images (1.5x)
2. **Pixel-level weighting**: Class weights calculated from training data (54.22 for pneumothorax, 0.50 for background)
3. **Loss function**: Combined Dice + Cross-Entropy loss to handle both sample and pixel imbalance

### Training Progression

The model shows steady improvement over 100+ epochs:
- Early epochs (1-20): Dice 0.35-0.48
- Mid epochs (20-50): Dice 0.48-0.58
- Late epochs (50-100): Dice 0.58-0.61

## Results

### Precision-Recall Curve
The model achieves **AP = 0.7030** with optimal F1-score at threshold 0.62.

### Sample Predictions
The model successfully segments pneumothorax regions in chest X-rays, with particularly good performance on larger, well-defined pneumothorax cases.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{upernet-pneumothorax,
  author = {Daksh Prajapati},
  title = {UPerNet Pneumothorax Segmentation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/upernet-pneumothorax}
}
```

## Acknowledgments

- [SIIM-ACR Pneumothorax Segmentation Challenge](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [UPerNet](https://arxiv.org/abs/1807.10221)

## License

MIT License - See LICENSE file for details.
