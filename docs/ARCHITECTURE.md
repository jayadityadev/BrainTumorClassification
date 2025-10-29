# System Architecture (updated)

Overview
--------
This repo implements a small, production-style pipeline for brain MRI tumor classification. The primary components are:

- Preprocessing (convert + enhancement)
- Dataset orchestration (combine .mat CE‑MRI + Kaggle images)
- Training (transfer learning: DenseNet121, ResNet50)
- Inference (predict + Grad-CAM visualization)
- Web app (Flask) for uploads and visualization

Key run-time stats (verified)
----------------------------
- Combined training samples: 6,568
- Combined test samples: 1,519
- Best model (DenseNet121): `models/current/densenet121/densenet121_final_20251029_215941.keras`
- DenseNet121 test accuracy: 98.49%
- ResNet50 test accuracy: 96.58%
- Inference latency (avg): ~51 ms / image

Module map (where to look)
---------------------------
- Preprocessing: `src/preprocessing/convert_mat_to_png.py`, `src/preprocessing/enhance.py`
- Dataset combine: `src/data/combine_datasets.py`
- Training: `src/models/train_combined_dataset.py`, `src/models/fast_finetune_kaggle.py`
- Inference & explainability: `src/inference/predict.py`, `src/inference/gradcam.py`
- App & UI: `app.py`, `templates/`

Data flow
---------
1. Raw CE‑MRI (.mat) files are converted to PNG via `convert_mat_to_png.py`.
2. Images are enhanced with `enhance.py` (Non-local means denoising + CLAHE + normalization).
3. Kaggle images are enhanced similarly and combined via `combine_datasets.py` to produce train/test CSVs in `data/combined_data_splits/`.
4. Training scripts create models saved under `models/current/<model_name>/` (phase checkpoints, final model, confusion matrix and history images).
5. Inference pipeline preprocesses input → model.predict → Grad-CAM → 3‑panel visualization saved to `outputs/predictions/`.

Model details
-------------
- DenseNet121 (transfer learned)
  - Input: (128, 128, 3)
  - Trainable parameters: ~7.3M
  - Final model path (from run): `models/current/densenet121/densenet121_final_20251029_215941.keras`

- ResNet50 (alternate)
  - Trainable parameters: ~24M
  - Final model path: `models/current/resnet50/resnet50_final_20251029_222554.keras`

Operational notes
-----------------
- Preprocessing (enhance.py) is critical for top performance. The validation run explicitly called this and the final models depend on the enhanced images.
- GPU memory: the verified run used an NVIDIA GeForce GTX 1650 with ~2.6GB available in TensorFlow; the fine‑tuning phase produced allocator OOM warnings but completed. If you run on small GPUs, lower batch size or use mixed precision where appropriate.
- The repo includes `scripts/validate_system.py` which performs a 10‑test smoke validation (GPU, model loading, prediction, enhancement checks, and performance benchmarks).

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

Last updated: auto-generated from run on 2025-10-29
