# 📘 Day 3 Notebooks - CNN Model Setup

**Dataset:** Combined Enhanced Dataset (7,181 images from 510 patients)  
**Goal:** Prepare data splits, configure augmentation, build CNN model, validate pipeline

---

## 📚 Notebooks Overview

### **1. day3_01_data_splitting.ipynb** (20 minutes)

**Purpose:** Create patient-wise train/val/test splits

**What it does:**
- ✅ Loads `metadata_enhanced.csv` (7,181 enhanced images)
- ✅ Groups images by patient ID (510 patients)
- ✅ Uses `StratifiedGroupKFold` to maintain class balance
- ✅ Ensures zero patient leakage between splits
- ✅ Creates train_split.csv, val_split.csv, test_split.csv

**Outputs:**
- `outputs/data_splits/train_split.csv` (~4,811 images, 67%)
- `outputs/data_splits/val_split.csv` (~861 images, 12%)
- `outputs/data_splits/test_split.csv` (~1,509 images, 21%)
- `outputs/visualizations/day3_01_*.png`

**Key Concept:** Patient-wise splitting prevents data leakage (no patient in both train and test)

---

### **2. day3_02_data_augmentation.ipynb** (15 minutes)

**Purpose:** Configure data augmentation pipeline

**What it does:**
- ✅ Loads train/val/test splits from notebook 1
- ✅ Configures ImageDataGenerator for training
- ✅ Visualizes augmented samples
- ✅ Saves augmentation configuration
- ✅ Creates data generators for training

**Augmentation Techniques:**
```python
- Rotation: ±15°
- Width/Height Shift: 5%
- Zoom: ±10%
- Horizontal Flip: Yes
- Vertical Flip: No (unsafe for brain MRI)
```

**Outputs:**
- `outputs/configs/augmentation_config.json`
- `outputs/visualizations/day3_02_*.png`
- Data generators ready for training

**Key Concept:** Safe medical image augmentation (preserves diagnostic meaning)

---

### **3. day3_03_cnn_architecture.ipynb** (15 minutes)

**Purpose:** Design and compile CNN model

**What it does:**
- ✅ Builds CNN architecture from scratch
- ✅ Analyzes model complexity (~4.29M parameters)
- ✅ Visualizes architecture diagram
- ✅ Compiles model with Adam optimizer
- ✅ Saves model configuration

**Model Architecture:**
```
Input (128×128×1) 
  ↓
Conv2D (32) + MaxPool
  ↓
Conv2D (64) + MaxPool
  ↓
Conv2D (128) + MaxPool
  ↓
Flatten + Dense (128) + Dropout (0.5)
  ↓
Output (3 classes, Softmax)
```

**Outputs:**
- `outputs/configs/model_architecture.json`
- `outputs/configs/model_summary.txt`
- `outputs/visualizations/day3_03_*.png`

**Key Concept:** Balanced model size (not too shallow, not too deep)

---

### **4. day3_04_training_test.ipynb** (10 minutes)

**Purpose:** Validate complete pipeline with 3-epoch test

**What it does:**
- ✅ Loads data generators from notebook 2
- ✅ Loads CNN model from notebook 3
- ✅ Runs 3-epoch training test
- ✅ Plots training curves
- ✅ Makes sample predictions
- ✅ Validates pipeline works correctly

**Expected Results:**
- Training accuracy: ~60-70% (3 epochs only)
- Validation accuracy: ~70-80% (good generalization!)
- Training time: ~10-15 seconds per epoch (GPU)

**Outputs:**
- `outputs/configs/day3_test_training_history.json`
- `outputs/visualizations/day3_04_*.png`
- Confidence that everything works!

**Key Concept:** Short test run to validate before full training

---

## 🚀 How to Run

### **Sequential Execution (Recommended)**

```bash
# 1. Start Jupyter
jupyter notebook notebooks/day3/

# 2. Run notebooks in order:
# - day3_01_data_splitting.ipynb
# - day3_02_data_augmentation.ipynb
# - day3_03_cnn_architecture.ipynb
# - day3_04_training_test.ipynb

# 3. Each notebook builds on the previous one
```

### **Quick Validation**

```bash
# Check if ready to run
ls outputs/data_splits/metadata_enhanced.csv  # Should exist
ls outputs/ce_mri_enhanced/                   # Should have images
ls outputs/ce_mri_enhanced_kaggle/            # Should have images

# After running notebook 1, verify splits
wc -l outputs/data_splits/*_split.csv

# After notebook 4, verify all configs
ls outputs/configs/
```

---

## 📊 Expected Timeline

| Notebook | Time | Difficulty | Outputs |
|----------|------|------------|---------|
| 3.1 Data Splitting | 20 min | Easy | 3 CSVs, 2 plots |
| 3.2 Augmentation | 15 min | Medium | Config, 6 plots |
| 3.3 Architecture | 15 min | Medium | Config, 3 plots |
| 3.4 Validation | 10 min | Easy | History, 2 plots |
| **Total** | **60 min** | | **15+ files** |

---

## 🐛 Common Issues & Solutions

### Issue 1: "metadata_enhanced.csv not found"

**Solution:**
```bash
# Run Day 2 enhancement first
.venv/bin/python src/preprocessing/enhance_combined_dataset.py

# Update metadata
.venv/bin/python src/preprocessing/update_metadata_enhanced.py
```

### Issue 2: "No module named 'tensorflow'"

**Solution:**
```bash
source .venv/bin/activate
pip install tensorflow
```

### Issue 3: "Found 0 images" in data generators

**Solution:**
- Check that `filepath` column in CSVs has correct paths
- Verify enhanced images exist
- Check `label` column is string type: `df['label'].astype(str)`

### Issue 4: "GPU not found"

**Solution:**
- CPU training will work (slower)
- GPU is optional but recommended
- Check: `tensorflow.config.list_physical_devices('GPU')`

---

## 📈 What You'll Learn

### **Technical Skills:**
- ✅ Patient-wise data splitting with StratifiedGroupKFold
- ✅ Medical image augmentation best practices
- ✅ CNN architecture design principles
- ✅ Keras/TensorFlow model building
- ✅ Training pipeline validation

### **Machine Learning Concepts:**
- ✅ Data leakage prevention
- ✅ Stratified sampling
- ✅ Overfitting mitigation
- ✅ Model capacity vs. dataset size
- ✅ Validation strategy

### **Domain Knowledge:**
- ✅ Medical image constraints
- ✅ MRI scan variability
- ✅ Clinical deployment considerations
- ✅ Patient privacy (why patient-wise splits matter)

---

## 🎯 Success Criteria

After completing Day 3, you should have:

- [x] ✅ Train/val/test splits created (patient-wise)
- [x] ✅ Zero patient overlap verified
- [x] ✅ Data augmentation configured
- [x] ✅ CNN model built and compiled
- [x] ✅ 3-epoch test passed (>70% val accuracy)
- [x] ✅ All configs saved
- [x] ✅ 15+ visualizations generated
- [x] ✅ Ready for Day 4 full training!

---

## 📝 Next Steps

After Day 3:

1. **Review Results**
   - Check training curves
   - Verify class distribution in splits
   - Examine augmented samples

2. **Day 4: Full Training**
   - Train for 10-15 epochs
   - Use callbacks (EarlyStopping, ModelCheckpoint)
   - Evaluate on test set
   - Export final model

3. **Optional: Ablation Studies**
   - Compare original vs enhanced images
   - Try different augmentation parameters
   - Experiment with model architecture

---

## 📚 References

- **Keras ImageDataGenerator:** https://keras.io/api/preprocessing/image/
- **StratifiedGroupKFold:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- **Medical Image Augmentation:** Perez & Wang, "The Effectiveness of Data Augmentation in Image Classification"

---

**Questions?** Check `docs/PROJECT_STATUS.md` for current status or `QUICK_REFERENCE.md` for commands.

---

*Last Updated: October 24, 2025*
