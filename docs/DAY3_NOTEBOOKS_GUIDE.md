# 📚 Day 3 - CNN Model Setup

Welcome to Day 3 of the Brain Tumor Classification project!

## 🎯 Overview

Today's focus is on building and validating a complete deep learning pipeline for brain tumor classification using CNNs.

---

## 📓 Notebooks (Execute in Order)

### 1. `day3_01_data_splitting.ipynb` ⏱️ ~30 min
**What you'll learn:**
- Why patient-wise splitting prevents data leakage
- How to use StratifiedGroupKFold
- Creating reproducible train/val/test splits

**Output:** `train_split.csv`, `val_split.csv`, `test_split.csv`

---

### 2. `day3_02_data_augmentation.ipynb` ⏱️ ~45 min
**What you'll learn:**
- Data augmentation for medical images
- Safe vs. unsafe augmentation techniques
- Building Keras ImageDataGenerator pipelines

**Output:** Augmentation config + visualization of transformations

---

### 3. `day3_03_cnn_architecture.ipynb` ⏱️ ~45 min
**What you'll learn:**
- CNN fundamentals (Conv2D, MaxPooling, Dense layers)
- Designing models for your hardware
- Parameter calculation and model sizing

**Output:** Compiled CNN model (~4.29M parameters)

---

### 4. `day3_04_training_test.ipynb` ⏱️ ~1 hour
**What you'll learn:**
- End-to-end pipeline validation
- Running a short training test (3 epochs)
- Interpreting learning curves

**Output:** Training history, learning curves, sample predictions

---

##  Key Deliverables

### Data Files:
- ✅ Train/val/test splits (patient-wise, no leakage)
- ✅ Augmentation configuration
- ✅ Model architecture JSON

### Visualizations:
- ✅ Data distribution analysis
- ✅ Augmentation examples
- ✅ Model architecture diagrams
- ✅ Learning curves
- ✅ Sample predictions

---

## 🚀 Quick Start

```python
import sys
sys.path.append('../..')

from src.modeling.data_generator import create_train_generator, create_val_test_generator
from src.modeling.model_cnn import build_cnn_model

# Create generators
train_gen = create_train_generator()
val_gen = create_val_test_generator('../outputs/data_splits/val_split.csv')

# Build model
model = build_cnn_model()

# Train (short test)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3
)
```

---

## 📝 Learning Progression

### Beginner-Friendly Structure:
Each notebook is self-contained with:
- 🧠 **Theory sections** - Understand the "why"
- 💻 **Code cells** - Hands-on implementation
- 📊 **Visualizations** - See results immediately
- 🎓 **Key takeaways** - Summarize learnings

### Progressive Complexity:
1. Start with simple concepts (data splitting)
2. Build up to augmentation (data engineering)
3. Understand model architecture (deep learning)
4. Validate complete pipeline (integration)

---

## ⚙️ System Requirements

### Hardware:
- **GPU:** NVIDIA GTX 1650 or better (4GB+ VRAM)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** ~5GB for datasets + models

### Software:
```bash
# Key dependencies
pip install tensorflow pandas scikit-learn matplotlib seaborn
```

### CUDA Setup:
Ensure CUDA and cuDNN are installed for GPU acceleration. Check with:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## 🎓 Learning Objectives

By the end of Day 3, you will:
1. ✅ Understand patient-wise data splitting for medical ML
2. ✅ Know which augmentations are safe for medical images
3. ✅ Design CNN architectures from scratch
4. ✅ Validate complete training pipelines
5. ✅ Interpret learning curves and model performance
6. ✅ Write modular, reusable ML code

---

## 📈 Expected Results

### After Running All Notebooks:
- **Training pipeline:** Fully functional end-to-end
- **Model performance:** >33% accuracy (better than random) in 3 epochs
- **GPU utilization:** Confirmed CUDA acceleration
- **No errors:** Complete pipeline runs smoothly

### Success Criteria:
- ✅ All notebooks execute without errors
- ✅ Visualizations generated and saved
- ✅ CSV files created with correct splits
- ✅ Model compiles and trains successfully
- ✅ Learning curves show improving accuracy

---

## 🔍 Troubleshooting

### Common Issues:

**1. GPU not detected**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall TensorFlow with GPU support
pip install tensorflow[and-cuda]
```

**2. Out of memory error**
```python
# Enable memory growth in notebooks
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

**3. File not found errors**
- Check that Day 2 outputs exist: `outputs/ce_mri_enhanced/`
- Verify `metadata.csv` has `filepath` column
- Ensure working directory is `/notebooks/`

---

## 📚 Additional Resources

### Recommended Reading:
- [Keras ImageDataGenerator Documentation](https://keras.io/api/preprocessing/image/)
- [CNN Architectures for Medical Imaging](https://arxiv.org/abs/1702.05747)
- [Data Augmentation in Medical Imaging](https://link.springer.com/article/10.1186/s12880-019-0390-5)

### Next Steps:
After completing Day 3, you're ready for:
- **Day 4:** Full training (10-15 epochs) + comprehensive evaluation
- **Module 3:** Advanced techniques (transfer learning, ensemble methods)

---

## 🏆 Completion Checklist

Before moving to Day 4, ensure:
- [ ] All 4 notebooks executed successfully
- [ ] Train/val/test CSVs generated
- [ ] At least 11 visualizations saved in `outputs/visualizations/`
- [ ] 3-epoch test run completed with learning curves
- [ ] `docs/DAY3_COMPLETION_LOG.md` reviewed
- [ ] GPU utilization confirmed
- [ ] No patient leakage verified (zero overlap in splits)

---

## 💬 Questions?

If you encounter issues or have questions:
1. Check the completion log: `docs/DAY3_COMPLETION_LOG.md`
2. Review code comments in notebooks
3. Inspect Python module docstrings
4. Refer to the troubleshooting section above

---

**Happy Learning! 🚀🧠**

*Remember: Take breaks between notebooks. Understanding > speed!*

---

**Navigation:**
- ⬅️ [Day 2: Image Enhancement](../DAY2_COMPLETION_LOG.md)
- ➡️ [Day 4: Full Training & Evaluation](./day4_README.md)
- 🏠 [Project Home](../README.md)
