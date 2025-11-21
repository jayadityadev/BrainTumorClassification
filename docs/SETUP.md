## Setup & Installation — Ubuntu (Linux) and Windows

This is the **canonical setup guide** for the project. It provides compact, tested instructions for Linux (Ubuntu 20.04+) and Windows 10/11, with both GPU (NVIDIA) and CPU-only workflows.

---

### 🔎 At a Glance (Where to Look)

- **Just want to run it once?** Use the quick commands below or the End-to-End Run Guide in `README.md`.
- **Need OS-specific setup?** See **2. Ubuntu** and **3. Windows**.
- **Stuck on environment / GPU issues?** Jump to **6. Runtime checks** and **7. Troubleshooting**.
- **Deploying or optimizing?** See **11. Advanced / production notes**.

---

### ⚡ Summary (Quick Commands)

**Create venv, activate, install deps (GPU):**
```bash
git clone https://github.com/jayadityadev/BrainTumorClassification.git
cd BrainTumorClassification
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-gpu.txt   # or requirements-cpu.txt
```

**Create directories, download, preprocess, validate:**
```bash
python scripts/setup_directories.py
python scripts/download_datasets.py   # requires ~/.kaggle/kaggle.json for Kaggle dataset
python src/preprocessing/convert_mat_to_png.py
python src/preprocessing/enhance.py
python src/data/combine_datasets.py
python scripts/validate_system.py    # expected: 10/10 tests
```

---

### 1) Prerequisites

- **Python:** 3.11 recommended (3.10+ should work). Use the same interpreter for venv and installs.
- **Disk:** ~20 GB free (datasets + models + outputs).
- **Memory:** 8 GB minimum; 16 GB recommended.
- **GPU (optional):** NVIDIA GPU recommended; on smaller GPUs (e.g., GTX 1650) lower batch sizes may be required.

---

### 2) Ubuntu 20.04+ (Native Linux) — Step-by-Step

**a. System packages**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget unzip pkg-config libsndfile1
```

**b. Python + venv**
```bash
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

**c. GPU drivers (optional)**

- Install NVIDIA drivers via package manager or from NVIDIA site. Verify with `nvidia-smi`.
```bash
# Ubuntu convenience installer (may require reboot)
sudo ubuntu-drivers autoinstall
nvidia-smi
```
- CUDA/cuDNN: Follow the official TensorFlow GPU installation guide for the TF version you plan to use. Often the fastest path is to use the TF-provided binary (`pip install -r requirements-gpu.txt`) and ensure system CUDA and drivers are compatible.

**d. Install Python dependencies**
```bash
# GPU-enabled (recommended if you have a working NVIDIA driver + CUDA)
pip install -r requirements-gpu.txt

# CPU-only (if no GPU)
pip install -r requirements-cpu.txt
```

> **Note:** `requirements-gpu.txt` pins the main packages used by the project. If TensorFlow reports missing CUDA/cuDNN components, follow the official install instructions at <https://www.tensorflow.org/install/gpu> for matching CUDA/cuDNN versions.

---

### 3) Windows 10 / 11 — Step-by-Step

**a. Install Python**

- Download and install Python 3.11 from <https://www.python.org>. During install check **“Add Python to PATH”**.

**b. Create virtual environment (PowerShell)**
```powershell
# In PowerShell (run as non-admin in repo directory)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # or use activate.bat in cmd.exe
python -m pip install --upgrade pip setuptools wheel
```

**c. GPU support (optional)**

- Install NVIDIA driver, CUDA Toolkit and cuDNN as per TensorFlow's Windows GPU guide. Use NVIDIA's installers for drivers and CUDA. Then install the GPU dependencies via pip:
```powershell
pip install -r requirements-gpu.txt
```
- If you run into wheel/compatibility issues on Windows, consider using the official TensorFlow packages and follow the TF GPU install docs.

**d. CPU-only**
```powershell
pip install -r requirements-cpu.txt
```

---

### 4) Optional: Conda Workflow (Alternative)

- If you prefer Conda, create an environment and install packages there:

```bash
conda create -n braintumor python=3.11
conda activate braintumor
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-gpu.txt   # or requirements-cpu.txt
```

> **Tip:** Prefer the repo’s `requirements-*.txt` pip lists for reproducibility.

---

### 5) Project Setup (Common Across OS)

**a. Create required directories**
```bash
python scripts/setup_directories.py
```

**b. Download datasets (automated)**
```bash
python scripts/download_datasets.py
```
> **Note:** Kaggle download requires `~/.kaggle/kaggle.json` (Kaggle API credentials). The Figshare CE‑MRI archive downloader may take time and network bandwidth.

**c. Convert and enhance (critical)**
```bash
python src/preprocessing/convert_mat_to_png.py    # if you have .mat files
python src/preprocessing/enhance.py              # mandatory for top performance
python src/data/combine_datasets.py
```
> **Important:** The enhancement step (denoising + CLAHE + normalization) is **required** for the models to achieve the reported accuracy.

**d. Quick training (fast path)**
```bash
python src/models/fast_finetune_kaggle.py
```

**Full training:**
```bash
python src/models/train_combined_dataset.py
```

**e. Validate the system (smoke tests)**
```bash
python scripts/validate_system.py
```
Expected: 10/10 tests passing. A `validation_report.txt` is written to the repo root.

---

### 6) Verify Installation & Runtime Checks

**Check Python and pip:**
```bash
python --version
pip --version
```

**Check TensorFlow and GPU:**
```bash
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs', tf.config.list_physical_devices('GPU'))"
# Optional: on systems with NVIDIA drivers
nvidia-smi
```

**Quick inference smoke test (after model exists):**
```bash
python -c "from src.inference.predict import predict_with_localization; print('OK')"
```

---

### 7) Troubleshooting (Common Issues)

**GPU not detected / TF reports no GPU**

- Verify `nvidia-smi` shows a device and drivers are installed.
- Ensure the CUDA/cuDNN versions are compatible with the TensorFlow wheel you installed (see TF GPU install docs).
- If you installed `requirements-gpu.txt` but TF cannot find GPU, try installing the TF wheel recommended for your CUDA version.

**Out of memory (OOM) on GPU**

- Reduce `BATCH_SIZE` in training scripts (e.g., from 32 → 16 → 8).
- Enable memory growth in TF (example below) or run on CPU for small tests.

**Model file not found / cannot load model**

- Check `models/current/` for `densenet121` or `resnet50` folders and final `.keras` files.
- Re-run training or copy the model into `models/current/`.

**Permissions / path issues on Windows**

- Use PowerShell/CMD with appropriate user privileges and ensure paths do not contain non-ASCII characters.

---

### 8) Helpful Snippets

**Enable TF GPU memory growth (Python):**
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
```

**Check TF can use CUDA (Python):**
```python
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
```

---

### 9) Environment Variables (example `.env`)
```
FLASK_APP=app.py
FLASK_ENV=production
    MODEL_PATH=models/current/densenet121/densenet121_final_20251121_135727.keras
MODEL_TYPE=densenet121
HOST=0.0.0.0
PORT=5000
MAX_CONTENT_LENGTH=16777216
CUDA_VISIBLE_DEVICES=0
```

---

### 10) Success Criteria & Verification

After setup and running preprocessing and validation steps, you should see:

- `python scripts/validate_system.py` → **10/10 tests passed**
- Models saved to `models/current/<model_name>/` (check for final `.keras` file)
- Inference (GPU): ~50–100 ms per image on the hardware used in the verified run
- Grad-CAM visualizations saved to `outputs/predictions/`

If you need help diagnosing a specific failure, capture the output of `python scripts/validate_system.py` and `validation_report.txt` and open an issue with those logs.

---

### 11) Advanced / Production Notes (Short)

- **Containerization:** a Dockerfile skeleton exists; for GPU containers see NVIDIA's Docker + CUDA support and the `nvidia-docker` runtime.
- **Model optimization:** consider TF-TRT or ONNX/TensorRT for lower latency in production.
- **Serving:** use Gunicorn + Nginx or a dedicated model server for scaling.

---

Last updated: 2025-11-21
