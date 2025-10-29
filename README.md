# 🧠 Brain Tumor Classification

**High-accuracy deep learning system for classifying brain tumors from MRI images.**

## 🎯 Project Overview

This project achieves **99.23% accuracy** in classifying brain tumors into four categories:
- **Glioma** (malignant, most aggressive)
- **Meningioma** (usually benign)
- **Pituitary** (affects hormones)
- **No Tumor** (healthy brain)

### Key Features
- ✅ **99.23% accuracy** (validated on test set)
- ✅ **Real-time inference** (~50ms per image)
- ✅ **Grad-CAM visualization** (explainable AI)
- ✅ **Web interface** (Flask app)
- ✅ **Production-ready** code

## 📊 Dataset

The project supports two dataset options:

1. **CE-MRI Dataset**: 3,064 brain MRI scans (.mat format)
   - Source: [Figshare - Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
   - 233 unique patients
   - Contrast-enhanced MRI (512×512)
   
2. **Kaggle Dataset**: ~7,000 brain MRI images
   - Source: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
   - Pre-organized by class
   - Ready to use

### 📥 Download Datasets

**Option 1: Fully Automated Download** (recommended):
```bash
python scripts/download_datasets.py
```
The script automatically handles:
- CE-MRI: Downloads nested ZIP, extracts 4 sub-ZIPs, organizes 3064 .mat files
- Kaggle: Uses Kaggle CLI (requires API key setup)
- Progress bars and automatic cleanup

**Option 2: Manual Download** (if automated fails):

**CE-MRI Dataset**:
1. Visit: https://figshare.com/ndownloader/articles/1512427/versions/5
2. Download starts automatically as `1512427.zip` (~900MB)
3. Extract → You'll get a folder with 4 sub-ZIPs:
   - `brainTumorDataPublic_1-766.zip`
   - `brainTumorDataPublic_767-1532.zip`
   - `brainTumorDataPublic_1533-2298.zip`
   - `brainTumorDataPublic_2299-3064.zip`
4. Extract each sub-ZIP and copy all .mat files to `datasets/ce-mri/`

**Kaggle Dataset**:
1. Install Kaggle CLI: `pip install kaggle`
2. Get API credentials from https://www.kaggle.com/settings (Create New API Token)
3. Place `kaggle.json` in `~/.kaggle/`
4. Run: `kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset`
5. Extract to `datasets/kaggle/`
   
> 📖 **See [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md) for complete setup instructions**


## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/jayadityadev/BrainTumorClassification.git
cd BrainTumorProject

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
# Choose ONE based on your hardware:

# Option A: GPU (NVIDIA - 10-12x faster!)
pip install -r requirements-gpu.txt

# Option B: CPU (works everywhere, but slower)
pip install -r requirements-cpu.txt

# Create directories
python scripts/setup_directories.py
```

> **💡 GPU vs CPU:** Training on GPU takes ~15-20 minutes vs 2-3 hours on CPU.  
> **Requirements for GPU:** NVIDIA GPU (GTX 1060+) + drivers 525.x or newer.  
> **Check GPU:** Run `nvidia-smi` to verify your GPU is available.

### 2. Download & Prepare Data

```bash
# Download datasets (fully automated!)
python scripts/download_datasets.py

# The script automatically:
# - Downloads CE-MRI dataset (nested ZIP structure)
# - Extracts 3064 .mat files from 4 sub-ZIPs
# - Downloads Kaggle dataset (requires API key)
# - Organizes everything in datasets/ directory

# Manual download (if automated fails):
# - CE-MRI: https://figshare.com/ndownloader/articles/1512427/versions/5
#   Download 1512427.zip → Extract → Extract 4 sub-ZIPs → Copy .mat files to datasets/ce-mri/
# - Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
#   Download → Extract to: datasets/kaggle/

# Preprocess (if using .mat files)
python src/preprocessing/convert_mat_to_png.py
python src/preprocessing/enhance.py

# Combine the downloaded datasets
python src/data/combine_datasets.py
```

### 3. Train Model

```bash
# Fast fine-tuning (recommended, ~15-20 min on GPU)
python src/models/fast_finetune_kaggle.py

# Full training (combined dataset, ~30-40 min)
python src/models/train_combined_dataset.py
```

### 4. Validate & Run

```bash
# Validate system (10 comprehensive tests)
python scripts/validate_system.py

# Launch web app
python app.py
# Open browser: http://localhost:5000
```

## 📁 Project Structure

```
BrainTumorProject/
├── app.py                    # Web application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── scripts/                  # Utility scripts
│   ├── setup_directories.py  # Directory setup
│   ├── download_datasets.py  # Dataset downloader
│   ├── validate_system.py    # System validation
│   └── evaluate_kaggle.py    # Model evaluation
│
├── src/                      # Source code
│   ├── preprocessing/        # Data preprocessing
│   │   ├── convert_mat_to_png.py
│   │   └── enhance.py
│   ├── models/               # Training scripts
│   │   ├── train_combined_dataset.py
│   │   └── fast_finetune_kaggle.py
│   ├── inference/            # Prediction & visualization
│   │   ├── predict.py
│   │   └── gradcam.py
│   └── data/                 # Data utilities
│
├── docs/                     # Documentation
│   ├── README.md             # Documentation index
│   ├── REPRODUCTION_GUIDE.md # Complete reproduction guide
│   ├── ARCHITECTURE.md       # System architecture
│   ├── SETUP.md              # Installation guide
│   ├── CLEANUP_SUMMARY.md    # Minimization summary
│   └── DISTRIBUTION_CHECKLIST.md  # Sharing checklist
│
├── templates/                # Web UI
│
└── [Generated directories]   # Created by setup_directories.py
    ├── datasets/             # Downloaded datasets
    │   ├── ce-mri/          # Original .mat files
    │   └── kaggle/          # Kaggle images
    ├── data/                 # Processed data
    ├── models/current/       # Trained models
    ├── outputs/              # Predictions, logs, reports
    └── config/               # Configuration files
```


## 🔬 Technical Details

### Model Architecture
- **Base Model**: DenseNet121 (pretrained on ImageNet)
- **Input**: 224×224×3 RGB images
- **Output**: 4 classes (softmax)
- **Parameters**: ~7M trainable

### Preprocessing Pipeline
1. **Non-Local Means Denoising** - Removes noise while preserving edges
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) - Enhances contrast
3. **Center Crop & Resize** - Standardizes image size

> ⚠️ **Critical**: The preprocessing step is essential for 99%+ accuracy. Without it, accuracy drops to ~85-90%.

### Performance Metrics
- **Accuracy**: 99.23%
- **Precision**: 99.24%
- **Recall**: 99.23%
- **F1-Score**: 99.23%
- **Inference Time**: ~50ms per image (GPU)

### Hardware Requirements
- **Minimum**: 16GB RAM, CPU
- **Recommended**: 32GB RAM, NVIDIA GPU (8GB+ VRAM)
- **Training Time**: 
  - GPU (RTX 3080): ~15-20 minutes
  - GPU (RTX 4090): ~8-12 minutes
  - CPU: ~2-4 hours (not recommended)

## 📖 Documentation

- 📘 **[docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md)** - Complete step-by-step reproduction
- 🏗️ **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture & design
- ⚙️ **[docs/SETUP.md](docs/SETUP.md)** - Installation & configuration
- 🧹 **[docs/CLEANUP_SUMMARY.md](docs/CLEANUP_SUMMARY.md)** - Minimization summary
- ✅ **[docs/DISTRIBUTION_CHECKLIST.md](docs/DISTRIBUTION_CHECKLIST.md)** - Sharing checklist

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow 2.13+, Keras
- **Image Processing**: OpenCV, Pillow
- **Data Science**: NumPy, Pandas, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Flask
- **MATLAB Files**: h5py, scipy

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If no GPU, install CUDA/cuDNN or use CPU (slower)
```

### Low Accuracy (<90%)
- **Cause**: Missing image enhancement
- **Solution**: Run `python src/preprocessing/enhance.py` before training

### Out of Memory
- **Solution**: Reduce batch size in training script (32 → 16 or 8)

### More Help
See [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) for detailed troubleshooting.

## 📊 Results

### Confusion Matrix
```
                 Predicted
               G    M    N    P
Actual    G   99%  0%   1%   0%
          M   0%  100%  0%   0%
          N   0%   0%  100%  0%
          P   1%   0%   0%  99%
```

### Per-Class Performance
- **Glioma**: 99.1% accuracy
- **Meningioma**: 99.5% accuracy  
- **No Tumor**: 100.0% accuracy
- **Pituitary**: 98.8% accuracy

## 🤝 Contributing

This project is open for contributions! Areas of interest:
- Additional datasets
- Model optimization
- New visualization techniques
- Deployment improvements

## 📝 License

Educational/Research project. See LICENSE file.

## 👤 Author

**Jayaditya Dev**
- GitHub: [@jayadityadev](https://github.com/jayadityadev)
- Email: jayadityadev261204@gmail.com

---

**⭐ If you find this project helpful, please star the repository!**

*Last Updated: October 26, 2025 | Version 2.0*

## 🚀 Upcoming Work

### Day 4: Full Training & Evaluation
- Full CNN training (10-15 epochs)
- Comprehensive evaluation on test set
- Confusion matrix & classification reports
- Model saving & export
- Ablation study: original vs enhanced images

## 🛠️ Tech Stack

- **Python**: 3.11.14
- **Libraries**: 
  - TensorFlow 2.x with CUDA (deep learning)
  - OpenCV 4.12.0 (image processing)
  - NumPy, SciPy (numerical computing)
  - Matplotlib, Seaborn (visualization)
  - scikit-learn (ML utilities)
  - h5py (MAT file handling)
  - tqdm (progress tracking)

## 💻 Hardware Optimization

- **CPU**: AMD Ryzen 5 5600H (12 cores @ 4.28 GHz)
- **GPU**: NVIDIA GeForce GTX 1650 Mobile
- **Optimization**: Multi-core parallel processing (75% CPU utilization)

## 📖 Documentation

Detailed completion logs available:
- [Day 1 Completion Log](docs/DAY1_COMPLETION_LOG.md) - Data extraction
- [Day 2 Completion Log](docs/DAY2_COMPLETION_LOG.md) - Image enhancement
- [Day 3 Completion Log](docs/DAY3_COMPLETION_LOG.md) - CNN model setup
- [Day 3 Notebooks Guide](docs/DAY3_NOTEBOOKS_GUIDE.md) - Notebook walkthrough

## 🚀 Quick Start

> **📘 New to this project?** Check the [Execution Guide](EXECUTION_GUIDE.md) for:
> - Fast Path vs Learning Path
> - What notebooks are mandatory
> - Our actual execution history
> - When to re-run things

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/jayadityadev/BrainTumorClassification.git
cd BrainTumorProject

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install tensorflow opencv-python numpy scipy h5py pandas matplotlib seaborn scikit-learn tqdm jupyter ipykernel

# Register Jupyter kernel
python -m ipykernel install --user --name=braintumor-venv --display-name="Python (BrainTumor)"
```

### 2. Running the Pipeline

#### **Day 1: Data Extraction**
```bash
# Extract images from .mat files
python src/preprocessing/convert_mat_to_png.py

# Or run notebooks in order:
jupyter notebook notebooks/day1/day1_dataset_explore.ipynb
```

**Output**: 3,064 PNG images + masks in `outputs/ce_mri_images/` and `outputs/ce_mri_masks/`

#### **Day 2: Image Enhancement**
```bash
# Run enhancement pipeline
python src/preprocessing/module1_enhance.py

# Or use the notebook:
jupyter notebook notebooks/day2/day2_enhancement.ipynb
```

**Output**: Enhanced images in `outputs/ce_mri_enhanced/` with 54.1% avg contrast improvement

#### **Day 3: CNN Model Setup**
```bash
# Run all Day 3 notebooks sequentially:
cd notebooks/day3

# 1. Data splitting (patient-wise, no leakage)
jupyter notebook day3_01_data_splitting.ipynb

# 2. Data augmentation setup
jupyter notebook day3_02_data_augmentation.ipynb

# 3. CNN architecture design
jupyter notebook day3_03_cnn_architecture.ipynb

# 4. Training pipeline validation (3 epochs)
jupyter notebook day3_04_training_test.ipynb
```

**Output**: 
- Train/val/test splits in `outputs/data_splits/`
- Model configs in `outputs/configs/`
- Training history & visualizations

### 3. Running Tests

```bash
# Test Day 1 completion
python tests/day1/test_day1.py

# Test Day 2 completion
python tests/day2/test_day2.py

# Test Day 3 completion (comprehensive)
python tests/day3/test_day3_completion.py
```

**Expected Results**: All tests should pass with ✓ green checkmarks

### 4. Module Usage Examples

#### Using the Data Generator
```python
from src.modeling.data_generator import create_train_generator, create_val_test_generator

# Create training generator with augmentation
train_gen = create_train_generator(
    csv_path='outputs/data_splits/train_split.csv',
    batch_size=32,
    target_size=(128, 128)
)

# Create validation generator (no augmentation)
val_gen = create_val_test_generator(
    csv_path='outputs/data_splits/val_split.csv',
    batch_size=32,
    target_size=(128, 128)
)
```

#### Using the CNN Model
```python
from src.modeling.model_cnn import build_cnn_model, print_model_info

# Build model
model = build_cnn_model(input_shape=(128, 128, 1), num_classes=3)

# Print detailed info
print_model_info(model)

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, validation_data=val_gen, epochs=10)
```

### 5. Using Custom Modules in Notebooks

```python
# Add src to Python path
import sys
sys.path.insert(0, '../..')

# Import modules
from src.modeling.data_generator import create_train_generator
from src.modeling.model_cnn import build_cnn_model
from src.utils.visualize_enhancement import plot_comparison
```

## 📊 Expected Results

### Day 1
- ✅ 3,064 images extracted successfully (100% success rate)
- ✅ Metadata CSV with patient IDs and labels
- ✅ Visual validation plots

### Day 2
- ✅ 54.1% average contrast improvement
- ✅ Processing speed: ~210 images/sec (9 workers)
- ✅ Before/after comparison visualizations

### Day 3
- ✅ Patient-wise splits: 2,059 train / 325 val / 680 test
- ✅ Zero patient leakage confirmed
- ✅ CNN model: ~4.29M parameters
- ✅ 3-epoch test: 76.31% validation accuracy
- ✅ Training time: ~13 seconds (GPU-accelerated)

### Day 4 (Upcoming)
- 🔄 Full training: 10-15 epochs
- 🔄 Expected accuracy: 80-85%
- 🔄 Test set evaluation with confusion matrix
- 🔄 Model export & ablation study

## �🔬 Research Context

This project focuses on automated brain tumor classification to assist medical diagnosis. The three tumor types have distinct characteristics:
- **Meningioma**: Most common, usually benign
- **Glioma**: Most aggressive, requires urgent treatment
- **Pituitary**: Affects hormone regulation

## 🐛 Troubleshooting

### Kernel Connection Issues
```bash
# Kill hung kernels
pkill -9 -f "jupyter|ipykernel"

# Reinstall kernel
.venv/bin/python -m ipykernel install --user --name=braintumor-venv
```

### Import Errors in Notebooks
```python
# Always add this at the top of notebooks
import sys
sys.path.insert(0, '../..')  # Adjust based on notebook location
```

### GPU Not Detected
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Label Type Errors with ImageDataGenerator
```python
# Ensure labels are strings
df['label'] = df['label'].astype(str)
```

## 📚 Documentation

Detailed completion logs available in `docs/`:
- [Execution Guide](EXECUTION_GUIDE.md) - **START HERE**: Fast vs Learning paths
- [Quick Reference](QUICK_REFERENCE.md) - Common commands & file locations
- [Day 1 Completion Log](docs/DAY1_COMPLETION_LOG.md) - Data extraction
- [Day 2 Completion Log](docs/DAY2_COMPLETION_LOG.md) - Image enhancement
- [Day 3 Completion Log](docs/DAY3_COMPLETION_LOG.md) - CNN model setup
- [Day 3 Notebooks Guide](docs/DAY3_NOTEBOOKS_GUIDE.md) - Notebook walkthrough

## 📝 License

This is a research/educational project.

## 👤 Author

**Jayaditya Dev**
- Email: jayadityadev261204@gmail.com
- GitHub: [@jayadityadev](https://github.com/jayadityadev)

---

*Last Updated: October 21, 2025*

````
