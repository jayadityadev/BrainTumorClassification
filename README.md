# Brain Tumor Classification Project

A deep learning project for classifying brain tumors from MRI images into three categories: Meningioma, Glioma, and Pituitary tumors.

## 📊 Dataset

- **Total Images**: 3,064 brain MRI scans
- **Classes**: 3 tumor types
  - Class 1: Meningioma (708 images)
  - Class 2: Glioma (1,426 images)
  - Class 3: Pituitary (930 images)
- **Patients**: 233 unique patients
- **Format**: Contrast-enhanced MRI images (512×512 pixels)

## 🏗️ Project Structure

````markdown
# Brain Tumor Classification Project

A deep learning project for classifying brain tumors from MRI images into three categories: Meningioma, Glioma, and Pituitary tumors.

## 📊 Dataset

- **Total Images**: 3,064 brain MRI scans
- **Classes**: 3 tumor types
  - Class 1: Meningioma (708 images)
  - Class 2: Glioma (1,426 images)
  - Class 3: Pituitary (930 images)
- **Patients**: 233 unique patients
- **Format**: Contrast-enhanced MRI images (512×512 pixels)

## 🏗️ Project Structure

```
BrainTumorProject/
├── dataset/                          # Raw data (3,064 .mat files)
│   └── *.mat                        # Original MATLAB brain MRI files
│
├── outputs/                          # Generated outputs & artifacts
│   ├── ce_mri_images/               # Extracted PNG images (512×512)
│   ├── ce_mri_masks/                # Tumor segmentation masks
│   ├── ce_mri_enhanced/             # Enhanced images (Day 2)
│   ├── visualizations/              # Analysis plots & figures
│   ├── configs/                     # Configuration files
│   │   ├── augmentation_config.json
│   │   ├── model_architecture.json
│   │   └── model_summary.txt
│   ├── data_splits/                 # Train/val/test splits
│   │   ├── train_split.csv
│   │   ├── val_split.csv
│   │   ├── test_split.csv
│   │   ├── split_summary.csv
│   │   └── metadata.csv
│   ├── training_logs/               # TensorBoard & training logs
│   └── models/                      # Saved model checkpoints
│
├── src/                             # Source code modules
│   ├── preprocessing/               # Data preprocessing
│   │   ├── convert_mat_to_png.py   # MAT to PNG conversion
│   │   └── module1_enhance.py      # Image enhancement pipeline
│   ├── modeling/                    # Model architecture & training
│   │   ├── data_generator.py       # Data augmentation generators
│   │   └── model_cnn.py            # CNN architecture definitions
│   ├── utils/                       # Utility functions
│   │   └── visualize_enhancement.py # Visualization helpers
│   └── __init__.py
│
├── notebooks/                       # Jupyter notebooks (organized by day)
│   ├── day1/                       # Day 1: Data extraction
│   │   ├── day1_dataset_explore.ipynb
│   │   ├── day1_metadata.ipynb
│   │   ├── day1_visual_check.ipynb
│   │   └── day1_dataset_distribution_check.ipynb
│   ├── day2/                       # Day 2: Image enhancement
│   │   └── day2_enhancement.ipynb
│   ├── day3/                       # Day 3: CNN model setup
│   │   ├── day3_01_data_splitting.ipynb
│   │   ├── day3_02_data_augmentation.ipynb
│   │   ├── day3_03_cnn_architecture.ipynb
│   │   └── day3_04_training_test.ipynb
│   └── exploration/                # Experimental notebooks
│       └── test_kernel.ipynb
│
├── tests/                          # Test suites
│   ├── day1/
│   │   └── test_day1.py           # Day 1 validation tests
│   ├── day2/
│   │   └── test_day2.py           # Day 2 validation tests
│   ├── day3/
│   │   └── test_day3_completion.py # Day 3 completion tests
│   └── README.md
│
├── docs/                           # Documentation
│   ├── DAY1_COMPLETION_LOG.md     # Day 1 detailed report
│   ├── DAY2_COMPLETION_LOG.md     # Day 2 detailed report
│   ├── DAY3_COMPLETION_LOG.md     # Day 3 detailed report
│   ├── DAY3_NOTEBOOKS_GUIDE.md    # Day 3 notebooks guide
│   ├── PREPROCESSING_UPDATES.md
│   └── SETUP_VERIFICATION.md
│
├── .venv/                          # Virtual environment
├── .gitignore
└── README.md                       # This file
```
├── .venv/                          # Virtual environment
├── .gitignore
└── README.md                       # This file
```

## ✅ Completed Milestones

### Day 1: Data Extraction & Conversion ✅
- Converted 3,064 .mat files to PNG images
- Extracted tumor masks
- Generated metadata CSV with patient IDs and labels
- **Performance**: 100% success rate
- **Location**: `notebooks/day1/`, `src/preprocessing/convert_mat_to_png.py`

### Day 2: Image Enhancement ✅
- Implemented 3-stage enhancement pipeline:
  1. **Non-Local Means Denoising** (h=10)
  2. **CLAHE** (clipLimit=2.0, tileGridSize=8×8)
  3. **Normalization** [0, 255]
- Multi-core batch processing (9 workers, ~210 images/sec)
- **Results**: 54.1% average contrast improvement
  - Class 1: +52.0%
  - Class 2: +56.9%
  - Class 3: +53.4%
- Visual validation with comparison plots
- **Location**: `notebooks/day2/`, `src/preprocessing/module1_enhance.py`

### Day 3: CNN Model Setup ✅
- Patient-wise data splitting (67% train, 12% val, 21% test)
- Zero patient leakage confirmed
- Data augmentation pipeline (rotation ±15°, shift 5%, zoom 10%, h-flip)
- CNN architecture designed (~4.29M parameters)
- Complete pipeline validated with 3-epoch test
- **Results**: 76.31% validation accuracy in 13.2 seconds
- **Deliverables**:
  - 4 educational notebooks
  - 2 reusable Python modules (`data_generator.py`, `model_cnn.py`)
  - Train/val/test splits with metadata
  - 10+ visualizations
- **Location**: `notebooks/day3/`

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
