# Brain Tumor Classification - Quick Reference Guide

## 📁 Project Organization

### Root Level
```
BrainTumorProject/
├── dataset/              # Raw MATLAB files (3,064 .mat files)
├── outputs/              # All generated data and artifacts
├── src/                  # Source code modules
├── notebooks/            # Jupyter notebooks (organized by day)
├── tests/                # Test suites for validation
├── docs/                 # Documentation and completion logs
├── .venv/                # Python virtual environment
├── README.md             # Main project documentation
├── EXECUTION_GUIDE.md    # How to run the project (Fast vs Learning paths)
└── QUICK_REFERENCE.md    # This file
```

### Detailed Structure

#### 📂 **outputs/** - Generated Data & Artifacts
```
outputs/
├── ce_mri_images/        # 3,064 extracted PNG images (organized by class)
├── ce_mri_masks/         # Tumor segmentation masks
├── ce_mri_enhanced/      # Enhanced images (Day 2)
├── configs/              # Configuration files
│   ├── augmentation_config.json
│   ├── model_architecture.json
│   ├── model_summary.txt
│   └── day3_test_training_history.json
├── data_splits/          # Train/val/test splits
│   ├── train_split.csv   # 2,059 images (67%)
│   ├── val_split.csv     # 325 images (12%)
│   ├── test_split.csv    # 680 images (21%)
│   ├── split_summary.csv
│   └── metadata.csv
├── visualizations/       # All generated plots and figures
├── training_logs/        # TensorBoard logs
└── models/               # Saved model checkpoints
```

#### 💻 **src/** - Source Code
```
src/
├── preprocessing/        # Data preprocessing modules
│   ├── convert_mat_to_png.py    # MAT to PNG conversion
│   └── module1_enhance.py       # Image enhancement pipeline
├── modeling/             # Model & training modules
│   ├── data_generator.py        # Data augmentation generators
│   └── model_cnn.py             # CNN architecture
└── utils/                # Utility functions
    └── visualize_enhancement.py # Visualization helpers
```

#### 📓 **notebooks/** - Jupyter Notebooks
```
notebooks/
├── day1/                 # Data extraction notebooks
│   ├── day1_dataset_explore.ipynb
│   ├── day1_metadata.ipynb
│   ├── day1_visual_check.ipynb
│   └── day1_dataset_distribution_check.ipynb
├── day2/                 # Image enhancement
│   └── day2_enhancement.ipynb
├── day3/                 # CNN model setup
│   ├── day3_01_data_splitting.ipynb
│   ├── day3_02_data_augmentation.ipynb
│   ├── day3_03_cnn_architecture.ipynb
│   └── day3_04_training_test.ipynb
└── exploration/          # Experimental notebooks
    └── test_kernel.ipynb
```

#### 🧪 **tests/** - Validation Tests
```
tests/
├── day1/
│   └── test_day1.py      # Day 1 validation
├── day2/
│   └── test_day2.py      # Day 2 validation
├── day3/
│   └── test_day3_completion.py  # Day 3 comprehensive tests
└── README.md             # Test documentation
```

#### 📚 **docs/** - Documentation
```
docs/
├── DAY1_COMPLETION_LOG.md      # Day 1 detailed report
├── DAY2_COMPLETION_LOG.md      # Day 2 detailed report
├── DAY3_COMPLETION_LOG.md      # Day 3 detailed report
├── DAY3_NOTEBOOKS_GUIDE.md     # Day 3 notebooks walkthrough
├── PREPROCESSING_UPDATES.md    # Enhancement pipeline updates
└── SETUP_VERIFICATION.md       # Environment setup validation
```

---

## 🚀 Common Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Deactivate
deactivate
```

### Running Modules
```bash
# Day 1: Extract images from MAT files
python src/preprocessing/convert_mat_to_png.py

# Day 2: Enhance images
python src/preprocessing/module1_enhance.py
```

### Running Notebooks
```bash
# Start Jupyter
jupyter notebook

# Navigate to specific day
# notebooks/day1/ - Data extraction
# notebooks/day2/ - Image enhancement
# notebooks/day3/ - CNN model setup
```

### Running Tests
```bash
# Test Day 1 completion
python tests/day1/test_day1.py

# Test Day 2 completion
python tests/day2/test_day2.py

# Test Day 3 completion (comprehensive)
python tests/day3/test_day3_completion.py

# Run all tests sequentially
python tests/day1/test_day1.py && \
python tests/day2/test_day2.py && \
python tests/day3/test_day3_completion.py
```

### Using Modules in Python/Notebooks
```python
# Import preprocessing modules
from src.preprocessing.convert_mat_to_png import convert_dataset
from src.preprocessing.module1_enhance import enhance_image

# Import modeling modules
from src.modeling.data_generator import create_train_generator
from src.modeling.model_cnn import build_cnn_model

# Import utilities
from src.utils.visualize_enhancement import plot_comparison
```

---

## 📊 Key Files & Their Purpose

| File Path | Purpose |
|-----------|---------|
| `outputs/data_splits/train_split.csv` | Training data paths & labels |
| `outputs/data_splits/val_split.csv` | Validation data paths & labels |
| `outputs/data_splits/test_split.csv` | Test data paths & labels |
| `outputs/configs/augmentation_config.json` | Data augmentation parameters |
| `outputs/configs/model_architecture.json` | CNN model configuration |
| `outputs/configs/day3_test_training_history.json` | Training metrics |
| `docs/DAY3_COMPLETION_LOG.md` | Day 3 results & analysis |
| `tests/day3/test_day3_completion.py` | Comprehensive validation |

---

## 🔍 Finding Things

### Find a specific file
```bash
# Find notebooks
find notebooks/ -name "*.ipynb"

# Find Python modules
find src/ -name "*.py"

# Find CSV files
find outputs/ -name "*.csv"
```

### Check outputs
```bash
# View data splits
ls -lh outputs/data_splits/

# View visualizations
ls -lh outputs/visualizations/

# View configs
ls -lh outputs/configs/
```

### View documentation
```bash
# Main README
cat README.md

# Day completion logs
cat docs/DAY*_COMPLETION_LOG.md

# Test documentation
cat tests/README.md
```

---

## 📈 Current Project Status

| Phase | Status | Location |
|-------|--------|----------|
| **Day 1**: Data Extraction | ✅ Complete | `notebooks/day1/`, `src/preprocessing/convert_mat_to_png.py` |
| **Day 2**: Image Enhancement | ✅ Complete | `notebooks/day2/`, `src/preprocessing/module1_enhance.py` |
| **Day 3**: CNN Model Setup | ✅ Complete | `notebooks/day3/` |
| **Day 4**: Full Training | 🔜 Pending | TBD |

### Results Summary
- **Images Extracted**: 3,064 (100% success)
- **Enhancement**: 54.1% avg contrast improvement
- **Data Splits**: 2,059 train / 325 val / 680 test
- **Patient Leakage**: ✅ Zero overlap confirmed
- **Model**: ~4.29M parameters
- **Initial Training**: 76.31% validation accuracy (3 epochs)

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Import errors in notebooks
```python
# Add to top of notebook
import sys
sys.path.insert(0, '../..')
```

**Issue**: Kernel not found
```bash
# Reinstall kernel
.venv/bin/python -m ipykernel install --user --name=braintumor-venv
```

**Issue**: GPU not detected
```python
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices('GPU'))
```

**Issue**: Test failures
```bash
# Run from project root
cd /projects/ai-ml/BrainTumorProject
python tests/day3/test_day3_completion.py
```

---

## 📞 Quick Help

**📚 Documentation**:
- **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - How to run the project (Fast vs Learning paths) 🚀
- **[README.md](README.md)** - Main project documentation
- **[tests/README.md](tests/README.md)** - Test suite documentation

**📋 Day-by-Day Guides**:
- [Day 1 Completion Log](docs/DAY1_COMPLETION_LOG.md) - Data extraction results
- [Day 2 Completion Log](docs/DAY2_COMPLETION_LOG.md) - Enhancement results
- [Day 3 Completion Log](docs/DAY3_COMPLETION_LOG.md) - CNN setup results
- [Day 3 Notebooks Guide](docs/DAY3_NOTEBOOKS_GUIDE.md) - How to use Day 3 notebooks

**🔗 Links**:
- **GitHub**: https://github.com/jayadityadev/BrainTumorClassification

---

*Last Updated: October 21, 2025*
