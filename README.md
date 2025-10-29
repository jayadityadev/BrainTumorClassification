# Brain Tumor Classification

High-accuracy deep-learning system to classify brain MRI scans into three tumor classes (Glioma, Meningioma, Pituitary) and visualize model localization with Grad-CAM.

Summary of the last successful run (from your logs)
- Combined training samples: 6,568
- Combined test samples: 1,519
- Best model: `models/current/densenet121/densenet121_final_20251029_215941.keras` (DenseNet121)
- DenseNet121 test accuracy: 98.49% (precision/recall/F1 ≈ 0.98)
- ResNet50 test accuracy: 96.58%
- Inference benchmark: ~51.3 ms / image (avg)
- Validation: `scripts/validate_system.py` → 10/10 tests passed

Why this repo
- Preprocessing (denoising + CLAHE) is required — accuracy drops significantly if skipped.
- Training scripts, evaluation, explainability (Grad-CAM) and a simple Flask UI are included.

Quick start (minimal)
1) Create and activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2) Install deps (pick one)
```bash
# GPU (recommended when available)
pip install -r requirements-gpu.txt

# CPU-only
pip install -r requirements-cpu.txt
```
3) Create directories and download data
```bash
python scripts/setup_directories.py
python scripts/download_datasets.py
```
4) Preprocess & combine
```bash
python src/preprocessing/convert_mat_to_png.py
python src/preprocessing/enhance.py
```

See `docs/SETUP.md` for installation and `docs/ARCHITECTURE.md` for design and dataflow.

- Run `python scripts/validate_system.py` → 10/10 tests passed
- Model accuracy similar to verified run (DenseNet121 test accuracy ≈ 98.5%)
- Grad-CAM visualizations generate and save in `outputs/predictions/`
- Web app runs at `http://localhost:5000` and accepts uploads
- Inference latency (on GPU) ~50–100ms per image

If any item fails, re-check preprocessing (`python src/preprocessing/enhance.py`), model path under `models/current/`, and environment/dependencies.


**⭐ If you find this project helpful, please star the repository!**

*Last Updated: October 29, 2025 | Version 3.0*

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
- `docs/SETUP.md` — install, config examples and validation

## 🚀 Quick Start

See `docs/SETUP.md` for the canonical quick-start, examples, configuration snippets and validation commands.

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

*Last Updated: October 21, 2025*