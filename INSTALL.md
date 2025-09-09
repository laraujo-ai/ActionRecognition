# Installation Guide

This document provides comprehensive instructions for installing and setting up the ActionRecognition system on NVIDIA Jetson platforms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Verification](#verification)
- [Usage Examples](#usage-examples)

## Prerequisites

### Hardware Requirements
- NVIDIA Jetson platform (Nano, Xavier NX, Xavier AGX, Orin series)
- Minimum 4GB RAM (8GB+ recommended for optimal performance)
- 16GB+ storage space available

### Software Requirements
- JetPack 5.0+ (JetPack 6.0+ recommended)
- Python 3.8+ (up to 3.12 depending on JetPack version)
- CUDA-enabled environment with NVIDIA drivers

### System Information Commands
To check your system configuration:

```bash
# Check Python version
python3 --version

# Check JetPack version
jetson_release

# Check CUDA availability
nvidia-smi
```

## Installation

Follow these steps to install the ActionRecognition system on your NVIDIA Jetson platform.

### Step 1: Create Virtual Environment (Recommended)

First, create and activate a virtual environment to isolate the project dependencies:

```bash
# Clone the repository
git clone <repository-url>
cd ActionRecognition

# Create virtual environment
python3 -m venv myenv

# Activate environment
source myenv/bin/activate
```

**Important**: Always activate your virtual environment before running the application:
```bash
source myenv/bin/activate
```

### Supported Configurations

| JetPack Version | Python Versions | ONNX Runtime Version |
|----------------|-----------------|---------------------|
| 6.0+           | 3.8, 3.9, 3.10, 3.11, 3.12 | 1.19.0 |
| 5.1.2          | 3.8, 3.9, 3.10, 3.11 | 1.18.0 |
| 5.1.1          | 3.8, 3.9, 3.10, 3.11 | 1.16.0 |
| 5.0            | 3.8           | 1.12.1 |

### Step 2: Install ONNX Runtime GPU

1. **Identify your configuration**:
   ```bash
   python3 --version  # Note the major.minor version (e.g., 3.8)
   jetson_release     # Note your JetPack version (e.g., 5.1.2)
   ```

2. **Download the appropriate wheel**:
   - Visit the [Jetson Zoo ONNX Runtime page](https://www.elinux.org/Jetson_Zoo#ONNX_Runtime)
   - Find the wheel matching your JetPack and Python versions from the table above
   - Download the `.whl` file to the project directory

3. **Install the wheel**:
   ```bash
   pip install ./onnxruntime_gpu-*.whl
   ```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Verification

After installation, verify that everything is working correctly:

### Check ONNX Runtime Installation

```bash
python3 -c "import onnxruntime as ort; print('ONNX Runtime version:', ort.__version__); print('Available providers:', ort.get_available_providers())"
```

Expected output should include `TensorrtExecutionProvider` and `CUDAExecutionProvider`.

### Check Core Dependencies

```bash
python3 -c "import cv2, pydantic, numpy; print('✅ Core dependencies installed successfully')"
```

### Test Model Loading

```bash
# Make sure your virtual environment is activated
source myenv/bin/activate

# Quick test with a sample video (if available)
python3 main.py --media_link ./sample_video.mp4 --model_type classifier --clips_length 2
```

## Usage Examples

**Remember**: Always activate your virtual environment before running the application:
```bash
source myenv/bin/activate
```

### Basic Video File Processing

```bash
# Process a local video file
python3 main.py \
  --media_link ./video.mp4 \
  --model_type classifier \
  --clips_length 2
```

### RTSP Stream Processing

```bash
# Process an RTSP stream with H.264 encoding
python3 main.py \
  --media_link 'rtsp://username:password@192.168.1.100/stream' \
  --model_type classifier \
  --stream_codec h264 \
  --clips_length 3
```

### Feature Extraction Mode

```bash
# Extract features instead of classification
python3 main.py \
  --media_link ./video.mp4 \
  --model_type feature_extractor \
  --clips_length 2
```

### Advanced Configuration

```bash
# H.265 stream with longer clips
python3 main.py \
  --media_link 'rtsp://camera_ip/stream' \
  --model_type classifier \
  --stream_codec h265 \
  --clips_length 5
```

### System Requirements Verification Script

Run this script to check if your system meets all requirements:

```bash
#!/bin/bash
echo "=== System Requirements Check ==="
echo "Python version: $(python3 --version)"
echo "JetPack version: $(jetson_release | grep JETPACK || echo 'jetson_release not found')"
echo "CUDA available: $(nvidia-smi > /dev/null 2>&1 && echo 'Yes' || echo 'No')"
echo "Available storage: $(df -h . | tail -1 | awk '{print $4}')"
echo "Available memory: $(free -h | grep '^Mem' | awk '{print $7}')"
```

## Additional Resources

- [NVIDIA Jetson Zoo](https://www.elinux.org/Jetson_Zoo) - Official NVIDIA wheel distributions
- [ONNX Runtime Documentation](https://onnxruntime.ai/) - ONNX Runtime official docs
- [JetPack Documentation](https://docs.nvidia.com/jetson/jetpack/) - NVIDIA JetPack documentation

---

**Note**: This installation guide is specifically designed for NVIDIA Jetson platforms. For other ARM64 or x86_64 systems, please refer to the standard ONNX Runtime installation procedures.