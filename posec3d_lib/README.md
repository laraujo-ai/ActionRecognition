# PoseC3D Library

A Python library for pose-based action recognition using YOLOv8 pose estimation and PoseC3D neural networks.

## Overview

This library provides a complete pipeline for recognizing human actions from video sequences:

1. **Pose Estimation** - Uses YOLOv8 to extract 17 keypoints from video frames
2. **Preprocessing** - Converts keypoints to 3D heatmap volumes with temporal information
3. **Action Recognition** - Uses PoseC3D ONNX model to classify actions from NTU RGB+D 60 dataset

## Structure

- `configs/` - Configuration files for model parameters and paths
- `models/` - Core model components (pose estimators, preprocessors, recognizers)
- `data/` - Data processing utilities and static files (label mappings)
- `utils/` - Helper utilities for video processing and visualization
- `weights/` - Pre-trained model weights (PoseC3D ONNX models)
- `demo_classification.py` - Example usage script

## Quick Start

```python
import providers as prov

# Load configuration
config = prov.get_config("configs/posec3d_inference.py")

# Initialize components
pose_estimator = prov.get_yolo_pose_estimator(config)
transforms_pipeline = prov.get_transformations_pipeline(config)
preprocessor = prov.get_preprocessor(transforms_pipeline)
model_engine = prov.get_model_engine(config)
recognizer = prov.get_recognizer(preprocessor, model_engine, config.LABEL_MAP_PATH)

# Run inference
pose_results = pose_estimator.inference_on_video(frame_paths)
data = {"pose_results": pose_results, "img_shape": (h, w)}
predicted_action = recognizer.inference(data)
```

## Dependencies

- ultralytics (YOLOv8)
- onnxruntime
- opencv-python
- numpy

## Action Classes

Supports 60 action classes from NTU RGB+D dataset including basic activities like "drink water", "eat meal", "sit down", "stand up", etc.