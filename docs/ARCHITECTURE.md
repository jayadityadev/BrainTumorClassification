# System Architecture (updated)

Overview
--------
This repo implements a small, production-style pipeline for brain MRI tumor classification. The primary components are:

- Preprocessing (convert + enhancement)
- Dataset orchestration (combine .mat CE‑MRI + Kaggle images)
- Training (transfer learning: DenseNet121, ResNet50)
- Inference (predict + Grad-CAM visualization)
- Web app (Flask) for uploads and visualization

High-level pipeline (conceptual)
-------------------------------

```text
Raw CE-MRI (.mat) & Kaggle JPGs
         │
         ▼
  [Preprocessing]
  - convert_mat_to_png.py
  - enhance.py  (denoise + CLAHE)
         │
         ▼
  Enhanced image folders
         │
         ▼
  [Dataset Orchestration]
  - combine_datasets.py  → train_split.csv / test_split.csv
         │
         ▼
  [Training]
  - train_combined_dataset.py (DenseNet121 / ResNet50)
         │
         ▼
  models/current/<model_name>/*.keras
         │
         ▼
  [Inference & Explainability]
  - predict.py + gradcam.py  → outputs/predictions/
         │
         ▼
  [Web App]
  - app.py (Flask) + templates/static
```

Key run-time stats (verified)
----------------------------
- Combined training samples: 6,568
- Combined test samples: 1,519
- Best model (DenseNet121): `models/current/densenet121/densenet121_final_20251121_135727.keras`
- DenseNet121 test accuracy: 99.21%
- ResNet50 test accuracy: 96.51%
- Inference latency (avg): ~51 ms / image

Module map (where to look)
---------------------------
- Preprocessing: `src/preprocessing/convert_mat_to_png.py`, `src/preprocessing/enhance.py`
- Dataset combine: `src/data/combine_datasets.py`
- Training (core): `src/models/train_combined_dataset.py`
- Training (optional / advanced): `src/models/fast_finetune_kaggle.py` – fast fine-tuning of an existing DenseNet121 model on the combined dataset
- Inference & explainability: `src/inference/predict.py`, `src/inference/gradcam.py`
- App & UI: `app.py`, `templates/`

Data flow
---------
1. **Ingestion:** raw CE‑MRI (.mat) files and Kaggle JPGs are ingested.
2. **Preprocessing:** CE‑MRI .mat files are converted to PNG via `convert_mat_to_png.py`; all images (CE‑MRI + Kaggle) are enhanced with `enhance.py`.
3. **Dataset build:** `combine_datasets.py` scans enhanced folders and produces `train_split.csv` and `test_split.csv` in `data/combined_data_splits/`.
4. **Training:** `train_combined_dataset.py` reads the CSVs, trains DenseNet121/ResNet50 and saves models + confusion matrices + history plots under `models/current/<model_name>/`.
5. **Inference:** `predict.py` loads the latest model, applies temperature scaling + Grad-CAM (via `gradcam.py`) and writes 3‑panel visualizations to `outputs/predictions/`.
6. **Serving:** `app.py` exposes a Flask UI that wraps the inference pipeline for browser-based uploads.

Model details
-------------
- DenseNet121 (transfer learned)
       - Input: (128, 128, 3)
       - Trainable parameters: ~7.3M
       - Final model path (from latest run): `models/current/densenet121/densenet121_final_20251121_135727.keras`

- ResNet50 (alternate)
       - Trainable parameters: ~24M
       - Final model path (from latest run): `models/current/resnet50/resnet50_final_20251121_142635.keras`

Operational notes
-----------------
- Preprocessing (enhance.py) is critical for top performance. The validation run explicitly called this and the final models depend on the enhanced images.
- GPU memory: the verified run used an NVIDIA GeForce GTX 1650 with ~2.6GB available in TensorFlow; the fine‑tuning phase produced allocator OOM warnings but completed. If you run on small GPUs, lower batch size or use mixed precision where appropriate.
- The repo includes `scripts/validate_system.py` which performs a 10‑test smoke validation (GPU, model loading, prediction, enhancement checks, and performance benchmarks).
- The repo also includes two **optional / advanced** utilities:
  - `src/models/fast_finetune_kaggle.py` – fast fine-tuning of an existing DenseNet121 model on the combined dataset
  - `scripts/evaluate_kaggle.py` – evaluates cross-domain generalization on a Kaggle-only test set and produces detailed reports/plots under `outputs/reports/`

Where artifacts are saved
------------------------
- `models/current/<model_name>/` — checkpoints, final model, confusion matrix and training history PNGs
- `data/combined_dataset/` — images used for combined training
- `data/combined_data_splits/` — `train_split.csv`, `test_split.csv`
- `outputs/predictions/` — saved prediction visualizations
- `validation_report.txt` — last validation run report

Performance & limitations
-------------------------
- Inference: ~50ms per image (GPU), ~200–500ms on CPU depending on hardware
- Memory: model loaded ~300MB, but training and fine-tuning will increase GPU VRAM usage
- Accuracy: the combined-run produced 98.49% test accuracy for DenseNet121 (see `models/current/densenet121/` for confusion matrix and history)

Extending the system
---------------------
- To add a new model: implement a script under `src/models/` that follows the existing training API and saves to `models/current/<new_model>/`.
- To change preprocessing: update `src/preprocessing/enhance.py` and re-run `src/data/combine_datasets.py` and `train_combined_dataset.py`.
- To add endpoints: update `app.py` and call `src/inference/predict.py` for prediction + localization.

Last updated: 2025-11-21
