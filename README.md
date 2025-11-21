# Brain Tumor Classification

High-accuracy deep-learning system to classify brain MRI scans into three tumor classes (Glioma, Meningioma, Pituitary) and visualize model localization with Grad-CAM.

Summary of the last successful run (from your logs)
- Combined training samples: 6,568
- Combined test samples: 1,519
- Best model: `models/current/densenet121/densenet121_final_20251121_135727.keras` (DenseNet121)
- DenseNet121 test accuracy: 99.21% (precision/recall/F1 ≈ 0.99)
- ResNet50 test accuracy: 96.51%
- Inference benchmark: ~51.3 ms / image (avg)
- Validation: `scripts/validate_system.py` → 10/10 tests passed

Why this repo
- Preprocessing (denoising + CLAHE) is required — accuracy drops significantly if skipped.
- Training scripts, evaluation, explainability (Grad-CAM) and a simple Flask UI are included.

---

## 🔁 End-to-End Run Guide (Windows & Linux)

The following commands show the full recommended flow to reproduce the main results.

### 1) Clone the repository and change directory

**Windows / PowerShell**
```powershell
git clone https://github.com/jayadityadev/BrainTumorClassification.git
cd BrainTumorClassification
```

**Linux / macOS (bash)**
```bash
git clone https://github.com/jayadityadev/BrainTumorClassification.git
cd BrainTumorClassification
```

### 2) Create and activate a virtual environment

**Windows / PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

**Linux / macOS (bash)**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3) Install dependencies (choose GPU or CPU)

**All platforms (inside venv)**
```bash
# GPU (recommended when you have a compatible NVIDIA GPU)
pip install -r requirements-gpu.txt

# or CPU-only (slower, but no GPU required)
# pip install -r requirements-cpu.txt
```

### 4) Create required directories

**Windows / PowerShell / Linux / macOS (bash)**
```bash
python scripts/setup_directories.py
```

### 5) Download datasets

**Windows / PowerShell / Linux / macOS (bash)**
```bash
# Preferred: Kaggle API configured with kaggle.json (see docs/SETUP.md)
python scripts/download_datasets.py
```

If you do **not** have `~/.kaggle/kaggle.json` set up, follow the manual download instructions in `docs/SETUP.md` (Kaggle + CE-MRI), then place the extracted folders where `scripts/download_datasets.py` expects them (e.g. `datasets/kaggle/`, `data/ce_mri_mat/`).

### 6) Convert CE-MRI .mat files to PNG (if you have CE-MRI dataset)

```bash
python src/preprocessing/convert_mat_to_png.py
```

### 7) Enhance images (MANDATORY)

```bash
python src/preprocessing/enhance.py
```

### 8) Combine enhanced datasets into CSV splits

```bash
python src/data/combine_datasets.py
```

### 9) Train on the combined dataset (DenseNet121 + ResNet50)

```bash
python src/models/train_combined_dataset.py
```

### 10) Validate the full system (10 tests)

```bash
python scripts/validate_system.py
```

If validation passes (10/10), you should see DenseNet121 test accuracy ≈ 98.5%, models under `models/current/`, and Grad-CAM-ready inference.

### Optional: Fast Kaggle-Only Fine-Tuning (Advanced)

`src/models/fast_finetune_kaggle.py` is an **optional / advanced** script for quickly fine-tuning an existing DenseNet121 model on the combined dataset (with a small learning rate and fewer epochs). It is useful when you already have a strong base model and just want to adapt it to new or updated Kaggle/combined data without rerunning the full training pipeline. It is **not required** for reproducing the main paper/README results.

Typical usage (after you have the CSV splits and an existing model at the configured `EXISTING_MODEL_PATH`):
```powershell
python src/models/fast_finetune_kaggle.py
```

This will:
- Load the existing DenseNet121 model
- Fine-tune it gently on the combined dataset
- Save the updated model and confusion matrix under `outputs/models/kaggle_finetuned/`.

---

**⭐ If you find this project helpful, please star the repository!**

---

## 📁 Project Structure (Key Files)

```text
BrainTumorClassification/
├── app.py                      # Flask web app for uploads + Grad-CAM visualization
├── src/
│   ├── preprocessing/
│   │   ├── convert_mat_to_png.py   # Convert CE-MRI .mat files to PNG images
│   │   └── enhance.py              # Denoising + CLAHE enhancement (critical for accuracy)
│   ├── data/
│   │   └── combine_datasets.py     # Build combined train/test CSVs from enhanced datasets
│   ├── models/
│   │   ├── train_combined_dataset.py  # Full training on combined CE-MRI + Kaggle data
│   │   └── fast_finetune_kaggle.py    # Optional fast fine-tuning of an existing DenseNet121
│   └── inference/
│       ├── predict.py             # Inference + temperature scaling + Grad-CAM wrapper
│       └── gradcam.py             # Grad-CAM utilities
├── scripts/
│   ├── setup_directories.py       # Create data/models/outputs directory skeleton
│   ├── download_datasets.py       # Download Kaggle + CE-MRI datasets (or guide manual steps)
│   ├── validate_system.py         # 10-test validation suite (GPU, data, model, app)
│   └── evaluate_kaggle.py         # Optional Kaggle-only evaluation and report generation
├── models/
│   └── current/                   # Saved DenseNet121/ResNet50 models + training artifacts
├── data/                          # Enhanced images + combined_data_splits CSVs
├── outputs/                       # Predictions, logs, reports
├── docs/                          # SETUP, ARCHITECTURE and other documentation
├── templates/                     # HTML templates for the Flask web UI
└── static/                        # CSS/JS assets for the web UI
```

This structure supports a simple four-stage pipeline: **convert → enhance → combine → train**, plus a Flask app and validation scripts built around those components.

*Last Updated: November 21, 2025 | Version 3.2*

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

Further documentation (key files):

- `docs/ARCHITECTURE.md` — architecture, dataflow and model locations
- `docs/SETUP.md` — detailed, OS-specific setup, troubleshooting and advanced configuration

## 🚀 Quick Start vs Full Setup

- Use the **End-to-End Run Guide** above for the minimal, happy-path sequence to reproduce results.
- Refer to `docs/SETUP.md` when you need **OS-specific instructions, environment debugging tips, or advanced setup** (GPU drivers, Conda, Docker notes, etc.).

## �🔬 Research Context

This project focuses on automated brain tumor classification to assist medical diagnosis. The three tumor types have distinct characteristics:
- **Meningioma**: Most common, usually benign
- **Glioma**: Most aggressive, requires urgent treatment
- **Pituitary**: Affects hormone regulation

## 🐛 Troubleshooting

### Import Errors
```python
# If you see `ModuleNotFoundError: No module named 'src'`,
# add project root to Python path in your script or test environment:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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

Further documentation (key files):

- `docs/ARCHITECTURE.md` — architecture, dataflow and model locations
- `docs/SETUP.md` — install, config examples and validation
- `docs/DISTRIBUTION_CHECKLIST.md` — release checklist

## 📝 License

This is a research/educational project.

## 👤 Author

**Jayaditya Dev**
- Email: jayadityadev261204@gmail.com
- GitHub: [@jayadityadev](https://github.com/jayadityadev)

---

*Last Updated: November 21, 2025*