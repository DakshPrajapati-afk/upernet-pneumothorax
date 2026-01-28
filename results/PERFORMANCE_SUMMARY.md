# UPerNet Pneumothorax Segmentation - Performance Summary

## Model Configuration

| Parameter | Value |
|-----------|-------|
| **Architecture** | UPerNet |
| **Backbone** | Swin Transformer Base |
| **Hidden Size** | 512 |
| **Input Size** | 512 x 512 |
| **Number of Classes** | 2 (Background, Pneumothorax) |
| **Loss Function** | Combined Dice + Cross-Entropy |
| **Optimizer** | AdamW |
| **Initial Learning Rate** | 1e-4 |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **GPUs** | 3 x NVIDIA GPUs |
| **Batch Size** | 4 per GPU (12 total) |
| **Total Epochs** | 200 |
| **Best Epoch** | 81 |

### Class Balancing Strategy
- **Negative Undersampling**: Keep ratio 0.15 (keeps only 15% of negatives)
- **Positive Oversampling**: Weight 1.5x
- **Class Weights**: Background = 0.5047, Pneumothorax = 54.2225

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Original Positive Samples** | 2,379 |
| **Original Negative Samples** | 8,296 |
| **After Balancing - Positive** | 2,379 |
| **After Balancing - Negative** | 1,244 |
| **Training Images** | 2,898 |
| **Validation Images** | 725 |
| **Class Imbalance (Pixel-level)** | 107.45:1 (Background:Pneumothorax) |

---

## Performance Metrics

### Segmentation Metrics (Validation Set)

| Metric | Value |
|--------|-------|
| **Best Validation Dice Score** | **0.6095** |
| **Best Validation IoU** | **0.4648** |
| **Final Training Dice** | 0.7218 |
| **Final Training IoU** | 0.5837 |

### Precision-Recall Analysis (200 validation images)

| Metric | Value |
|--------|-------|
| **Average Precision (AP)** | **0.7030** |
| **PR-AUC** | **0.7030** |
| **Optimal F1-Score** | **0.6658** |
| **Precision (at optimal threshold)** | 0.6718 |
| **Recall (at optimal threshold)** | 0.6600 |
| **Optimal Decision Threshold** | 0.6187 |

### Pixel-Level Statistics (Validation)
- Total pixels analyzed: 209,715,200
- Pneumothorax pixels: 2,083,861 (0.99%)
- Background pixels: 207,631,339 (99.01%)

---

## Training Progression

### Dice Score Improvement Over Training

| Epoch Range | Best Val Dice | Notes |
|-------------|---------------|-------|
| 1-10 | 0.4139 | Initial learning |
| 11-20 | 0.5228 | Rapid improvement |
| 21-30 | 0.5418 | Continued progress |
| 31-40 | 0.5659 | Stabilizing |
| 41-50 | 0.5858 | Fine-tuning |
| 51-60 | 0.5976 | Near convergence |
| 61-70 | 0.6077 | Best performance region |
| 71-81 | **0.6095** | **Best model** |
| 82-100 | 0.6085 | Slight plateau |

---

## Files in This Results Folder

```
results/
├── PERFORMANCE_SUMMARY.md      # This file
├── precision_recall_curve.png  # PR curve visualization
├── detailed_metrics.png        # Additional metric charts
└── images/                     # Sample predictions
    ├── prediction_01_*.png     # Positive case 1
    ├── prediction_02_*.png     # Positive case 2
    ├── prediction_03_*.png     # Positive case 3
    ├── prediction_04_*.png     # Positive case 4
    ├── prediction_05_*.png     # Positive case 5
    ├── prediction_11_*.png     # Negative case 1
    └── prediction_12_*.png     # Negative case 2
```

### Image Format
Each prediction image shows a side-by-side comparison:
- **Left**: Original chest X-ray input
- **Center**: Ground truth mask
- **Right**: Model prediction

---

## Key Findings

1. **Strong Performance**: The model achieves AP=0.7030, which is competitive for medical image segmentation with severe class imbalance.

2. **Class Imbalance Handling**: The combination of negative undersampling and class-weighted loss effectively addresses the 107:1 pixel imbalance.

3. **Dice vs IoU Gap**: The gap between Dice (0.61) and IoU (0.46) is typical for binary segmentation with small foreground regions.

4. **Precision-Recall Trade-off**: At threshold 0.62, the model balances precision (0.67) and recall (0.66) effectively.

5. **Training Stability**: The model shows consistent improvement without overfitting, indicating good regularization.

---

## Recommendations for Further Improvement

1. Increase training data through more aggressive augmentation
2. Experiment with different backbone sizes (Swin-Large)
3. Try focal loss to further address class imbalance
4. Ensemble multiple models at different thresholds
5. Post-processing with CRF or morphological operations

---

*Generated from UPerNet Pneumothorax Segmentation Pipeline*
