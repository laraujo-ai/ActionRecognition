# ActionRecognition System Documentation

## Overview

The ActionRecognition system is a real-time action recognition pipeline that processes video files or RTSP streams to identify human actions using pose-based analysis. It combines YOLOv8 pose estimation with PoseC3D action classification for efficient and accurate action recognition.

## Architecture

The system follows a modular, threaded architecture designed for scalability and performance:

```
Input Media → Media Processor → Clip Generation → Inference Pipeline → Action Results
     ↓              ↓                ↓                    ↓                ↓
Video/RTSP → Frame Extraction → Video Clips → Pose + Action → Classifications
```

## Core Components

### 1. Main Application (`main.py`)
Entry point that orchestrates the entire pipeline:
- Parses command line arguments
- Creates media processors and ML models
- Manages threaded processing
- Handles cleanup and resource management

### 2. Media Processing Library (`media_processing_lib/`)
Handles video input processing and clip generation:

**Key Classes:**
- `VideoProcessor`: Processes local video files (MP4, AVI, MOV)
- `StreamProcessor`: Handles RTSP streams with NVIDIA hardware acceleration
- `Clip`: Data model representing video segments with frame paths

**Features:**
- Memory-efficient disk-based frame storage
- Non-overlapping clip generation
- Configurable clip duration
- Automatic temporary file management

### 3. Inference Pipeline (`src/`)
Manages the ML inference workflow:

**Components:**
- `inference.py`: Threaded inference processing
- `factory.py`: Model and processor factory functions
- Two-stage processing: Pose estimation → Action recognition

### 4. PoseC3D Library (`posec3d_lib/`)
Complete pose-based action recognition framework:

**Models:**
- **Pose Estimator**: YOLOv8-pose for 17-keypoint detection
- **Action Recognizer**: PoseC3D for action classification
- **Feature Extractor**: PoseC3D variant for feature extraction

**Data Processing:**
- Pose transformation pipeline
- Heatmap generation from keypoints
- Temporal sequence processing
- Data formatting for model input

**Utilities:**
- Video processing and frame extraction
- Visualization and drawing functions
- Configuration parsing and management

## Processing Pipeline

### 1. Media Input
- **Video Files**: Local files processed sequentially
- **RTSP Streams**: Real-time stream processing with hardware acceleration

### 2. Frame Processing
```
Video → Frame Extraction → Temporary Storage → Clip Creation
```
- Frames saved as individual JPG files
- Memory-efficient approach prevents RAM overflow
- Configurable clip duration (default: 2 seconds)

### 3. Pose Estimation
```
Video Frames → YOLOv8 → 17 Keypoints → Confidence Scores
```
- Detects human poses in each frame
- Extracts 17 COCO-format keypoints per person
- Filters by confidence threshold

### 4. Action Recognition
```
Pose Keypoints → Heatmap Generation → PoseC3D → Action Classification
```
- Converts keypoints to 3D heatmap volumes
- Processes temporal sequences
- Classifies actions from NTU RGB+D 60 dataset

## Configuration

### Command Line Usage
```bash
python main.py \
  --media_link ./video.mp4 \
  --model_type classifier \
  --clips_length 2
```

### Parameters
- `--media_link`: Video file path or RTSP URL
- `--model_type`: "classifier" or "feature_extractor"
- `--stream_codec`: Codec for stream processing (default: h264)
- `--clips_length`: Clip duration in seconds (default: 2)

### Model Configuration
Located in `posec3d_lib/configs/posec3d_inference.py`:
- Transformation pipeline parameters
- Model paths (relative to project root)
- Processing thresholds and settings

## Dependencies

### Core Requirements
```
ultralytics      # YOLOv8 models
opencv-python    # Video processing
onnxruntime      # ML inference
pydantic         # Data validation
```

### Optional (for RTSP)
- GStreamer with NVIDIA plugins
- Hardware acceleration support

## Model Files

### Pose Estimation
- `yolov8n-pose.pt`: Lightweight pose model (8MB)
- Supports various YOLOv8 sizes (n/s/m/l/x)

### Action Recognition
- `posec3d.onnx`: Action classification model (8MB)
- `posec3d_features.onnx`: Feature extraction variant (8MB)
- `label_map_ntu60.txt`: Action class names

## Performance Considerations

### Memory Optimization
- **Disk-based storage**: Frames saved to temp files vs RAM
- **Bounded queues**: Prevents infinite memory growth
- **Cleanup management**: Automatic temp file removal

### Threading Architecture
- **Video Thread**: Frame extraction and clip creation
- **Inference Thread**: Pose estimation and action recognition
- **Main Thread**: Result processing and coordination

### Hardware Acceleration
- **NVIDIA GPU**: Hardware-accelerated video decoding (RTSP)
- **ONNX Runtime**: Optimized ML inference
- **OpenCV**: Efficient video processing

## Action Classes

The system recognizes 60 actions from the NTU RGB+D dataset:
- Basic activities: drink water, eat meal, sit down, stand up
- Gestures: clapping, waving, pointing
- Interactions: handshaking, hugging, kicking
- And 50+ additional action categories

## Troubleshooting

### Common Issues
1. **Segmentation Fault**: Usually memory/temp file cleanup issues
2. **Empty Queues**: Check thread lifecycle and poison pill handling
3. **Model Loading**: Verify ONNX files exist and paths are correct
4. **RTSP Issues**: Ensure GStreamer NVIDIA plugins installed

### Debug Mode
Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Memory Issues
- Reduce clip length: `--clips_length 1`
- Check temp directory cleanup
- Monitor queue sizes

## Development

### Extending the System
1. **New Media Sources**: Implement `IMediaProcessor` interface
2. **Custom Models**: Add to factory functions in `src/factory.py`
3. **Additional Actions**: Retrain PoseC3D with new datasets
4. **Performance Tuning**: Adjust queue sizes and thread counts

This system provides a complete, production-ready solution for real-time action recognition with support for both video files and live RTSP streams.