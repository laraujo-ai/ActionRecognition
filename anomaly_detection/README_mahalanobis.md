# Mahalanobis Anomaly Detection Training

A clean, modular training framework for Mahalanobis-based anomaly detection on pose-based action recognition features.

## Overview

This framework trains an anomaly detection model that:
1. Extracts features from videos using PoseC3D
2. Learns normal behavior distribution using Mahalanobis distance
3. Detects anomalies by measuring distance from normal distribution

## Quick Start

1. Configure your training in `train_config.yaml`

2. Train the model:
   ```bash
   python train_mahalanobis.py train_config.yaml
   ```

## Configuration

The training is controlled by a YAML configuration file. See `train_config.yaml` for a complete example.

### Key Configuration Sections:

- **dataset**: Data paths and categories
- **model**: PoseC3D model configuration and preprocessing
- **training**: Training parameters and algorithms
- **output**: Model saving and results output

## Output

The training produces:

1. **Trained Model** (`mahalanobis_model.pkl`):
   - StandardScaler for feature normalization
   - PCA transformer (if enabled)
   - Mean vector and covariance matrix
   - Anomaly detection threshold

2. **Training Results** (`training_results.json`):
   - Configuration used
   - Dataset summary
   - Model information
   - Evaluation metrics

## Architecture

### Modules:

- **`data_loader.py`**: Video data loading and feature extraction
- **`mahalanobis_trainer.py`**: Model training and evaluation
- **`train_mahalanobis.py`**: Main training script

### Key Features:

- **Configuration-Driven**: YAML-based configuration
- **Robust**: Proper error handling and validation
- **Flexible**: Supports various dataset structures and parameters
- **Reproducible**: Saves complete training state and results

## Usage Examples

### Basic Training
```bash
python train_mahalanobis.py train_config.yaml
```

### Custom Configuration
```yaml
dataset:
  base_path: "/path/to/your/dataset"
  normal_categories:
    - "walking"
    - "sitting"
  anomaly_categories:
    - "falling"
```

### Loading Trained Model
```python
from mahalanobis_trainer import TrainedModel

model = TrainedModel.load("./trained_models/mahalanobis_model.pkl")
print(model.get_model_info())
```

## Dataset Structure

Expected directory structure:
```
dataset_base_path/
├── train/
│   ├── normal_category_1/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   └── normal_category_2/
├── validation/
│   ├── normal_category_1/
│   ├── anomaly_category_1/
│   └── anomaly_category_2/
└── test/
    ├── normal_category_1/
    ├── anomaly_category_1/
    └── anomaly_category_2/
```