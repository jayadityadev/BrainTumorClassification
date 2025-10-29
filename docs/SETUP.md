# Setup Guide

Complete installation and setup instructions for the Brain Tumor Classification System.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: 3.11 or higher
- **RAM**: Minimum 8GB, recommended 16GB
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster inference
- **Disk Space**: ~10GB for dataset and models

### Software Dependencies
- Python 3.11+
- pip (Python package manager)
- Git
- CUDA Toolkit 11.8+ (if using GPU)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jayadityadev/BrainTumorClassification.git
cd BrainTumorClassification
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install tensorflow==2.20.0
pip install numpy pandas matplotlib seaborn
pip install opencv-python pillow
pip install flask werkzeug
pip install scikit-learn scipy
pip install tqdm
```

**GPU Support** (Optional but recommended):
```bash
# Install CUDA-enabled TensorFlow packages
pip install nvidia-cudnn-cu12
pip install nvidia-cublas-cu12

# Verify GPU detection
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### 4. Verify Installation

```bash
# Run system validation
python scripts/validate_system.py
```

Expected output: **10/10 tests passing**

## Configuration

### GPU Setup (NVIDIA)

If you have an NVIDIA GPU:

1. **Install NVIDIA Drivers**:
```bash
# Ubuntu
sudo ubuntu-drivers autoinstall

# Check driver version
nvidia-smi
```

2. **Install CUDA Toolkit** (11.8+):
```bash
# Follow instructions at:
# https://developer.nvidia.com/cuda-downloads
```

3. **Verify GPU Access**:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Model Setup

The current production model should be automatically available at:
```
models/current/densenet121_finetuned.keras
```

If missing, you can download it from the releases or retrain using:
```bash
python src/models/train.py
```

## Dataset Setup

### Option 1: Use Existing Processed Data

If you have the processed dataset:
```bash
# Dataset should be in:
data/splits/train_split.csv
data/splits/test_split.csv
```

### Option 2: Process Raw Data

If starting from raw .mat files:

1. Place .mat files in `data/raw/`
2. Run conversion:
```bash
python src/preprocessing/convert_mat_to_png.py
```
3. Combine datasets:
```bash
python scripts/combine_datasets.py
```

## Running the Application

### Web Interface

Start the Flask web application:

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

### Command Line Interface

For single image prediction:

```python
from src.inference.predict import predict_with_localization

results = predict_with_localization(
    image_path="path/to/mri_scan.jpg",
    model_path="models/current/densenet121_finetuned.keras",
    model_type="densenet121"
)

print(f"Prediction: {results['predicted_class']}")
print(f"Confidence: {results['confidence']:.2f}%")
```

### Evaluation

Evaluate model performance:

```bash
python scripts/evaluate_kaggle.py
```

## Development Setup

### IDE Configuration

**VS Code** (Recommended):
1. Install Python extension
2. Set interpreter to `.venv/bin/python`
3. Enable Pylance for type checking
4. Configure debugger for Flask

### Testing

Run unit tests:
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_inference.py
```

## Troubleshooting

### GPU Not Detected

**Problem**: TensorFlow not detecting GPU

**Solutions**:
1. Check NVIDIA drivers: `nvidia-smi`
2. Install CUDA packages:
   ```bash
   pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
   ```
3. Reload nvidia_uvm module:
   ```bash
   sudo rmmod nvidia_uvm
   sudo modprobe nvidia_uvm
   ```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Add project root to Python path:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
```

### Out of Memory

**Problem**: GPU out of memory during inference

**Solutions**:
1. Reduce batch size
2. Use CPU instead: Set `CUDA_VISIBLE_DEVICES=""`
3. Close other GPU applications

### Model Loading Errors

**Problem**: Cannot load model file

**Solutions**:
1. Verify model path is correct
2. Check model file exists and isn't corrupted
3. Ensure TensorFlow version matches (2.20.0)

## Performance Optimization

### GPU Acceleration

Enable GPU memory growth:
```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Batch Prediction

For multiple images:
```python
# Process images in batches
batch_size = 32
for batch in batches(images, batch_size):
    predictions = model.predict(batch)
```

### Model Optimization

Convert to TensorRT for faster inference:
```bash
# Requires TensorRT installation
python -m tf2onnx.convert --saved-model models/current/ --output model.onnx
```

## Production Deployment

### Using Gunicorn

For production, use a production-grade WSGI server:

```bash
# Install gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t brain-tumor-classifier .
docker run -p 5000:5000 brain-tumor-classifier
```

### Using Nginx

Configure Nginx as reverse proxy:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Environment Variables

Create `.env` file for configuration:

```bash
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=production
FLASK_DEBUG=0

# Model Configuration
MODEL_PATH=models/current/densenet121_finetuned.keras
MODEL_TYPE=densenet121

# Server Configuration
HOST=0.0.0.0
PORT=5000
MAX_CONTENT_LENGTH=16777216  # 16MB

# GPU Configuration
CUDA_VISIBLE_DEVICES=0  # Use first GPU, -1 for CPU only
```

## Validation Checklist

After setup, verify everything works:

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] GPU detected (if applicable)
- [ ] Model file exists and loads
- [ ] Dataset accessible
- [ ] Web app starts successfully
- [ ] Single prediction works
- [ ] Validation script passes all tests
- [ ] Evaluation runs without errors

Run the complete validation:
```bash
python scripts/validate_system.py
```

Expected: **10/10 tests passing** ✅

## Getting Help

### Documentation
- [README.md](../README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [API.md](API.md) - API reference
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - Prediction guide

### Support
- GitHub Issues: Report bugs or request features
- Documentation: Check docs/ directory
- Validation: Run `python scripts/validate_system.py`

## Next Steps

Once setup is complete:

1. **Test the web interface**: Open http://localhost:5000
2. **Try a prediction**: Upload an MRI scan
3. **Review documentation**: Read through docs/
4. **Explore notebooks**: Check notebooks/archive/ for experiments
5. **Run evaluation**: `python scripts/evaluate_kaggle.py`

---

**Last Updated**: October 26, 2025  
**Setup Time**: ~15-30 minutes  
**Difficulty**: Intermediate
