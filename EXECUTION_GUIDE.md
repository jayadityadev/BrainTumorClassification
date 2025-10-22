# Brain Tumor Classification - Execution Guide

## 🎯 Overview

This guide explains **two paths** to run this project:
1. **Fast Path** - Run Python scripts only (production mode)
2. **Learning Path** - Run Jupyter notebooks (educational/exploratory mode)

---

## 📊 Execution History (What We Actually Did)

### **Day 1: Data Extraction** (October 20, 2025)

**Goal**: Convert 3,064 .mat files to PNG images

**What we ran**:
```bash
# Option 1: Direct Python script (FAST - what we used)
python src/preprocessing/convert_mat_to_png.py
# - day1_metadata.ipynb              # Generated metadata CSV (REQUIRED)

# Option 2: Jupyter notebooks (SLOW - exploratory)
# - day1_dataset_explore.ipynb       # Explored dataset structure
# - day1_visual_check.ipynb          # Visual validation
# - day1_metadata.ipynb              # Generated metadata CSV
# - day1_dataset_distribution_check.ipynb  # Class distribution
```

**Output**: 
- `outputs/ce_mri_images/` - 3,064 PNG images
- `outputs/ce_mri_masks/` - 3,064 masks
- `outputs/data_splits/metadata.csv` - Image metadata

**Time**: ~5 minutes (script) vs ~30 minutes (all notebooks)

**Verdict**: ✅ We used the script. Notebooks are optional for learning.

---

### **Day 2: Image Enhancement** (October 21, 2025)

**Goal**: Apply 3-stage enhancement (Denoising + CLAHE + Normalization)

**What we ran**:
```bash
# Option 1: Direct Python script (FAST - what we used)
python src/preprocessing/module1_enhance.py

# Option 2: Jupyter notebook (MEDIUM - exploratory)
# - day2_enhancement.ipynb  # Interactive enhancement with visualizations
```

**Output**:
- `outputs/ce_mri_enhanced/` - 3,064 enhanced images
- `outputs/visualizations/enhancement_*.png` - Comparison plots

**Time**: ~15 seconds (script with multiprocessing) vs ~5 minutes (notebook)

**Verdict**: ✅ We used the script. Notebook is good for understanding the pipeline.

---

### **Day 3: CNN Model Setup** (October 21, 2025)

**Goal**: Split data, configure augmentation, design CNN, validate pipeline

**What we ran**:
```bash
# ALL 4 notebooks were run sequentially (REQUIRED for Day 3)
jupyter notebook notebooks/day3/day3_01_data_splitting.ipynb
jupyter notebook notebooks/day3/day3_02_data_augmentation.ipynb
jupyter notebook notebooks/day3/day3_03_cnn_architecture.ipynb
jupyter notebook notebooks/day3/day3_04_training_test.ipynb
```

**Why notebooks this time?**
- Day 3 is about **designing and testing** the model
- Notebooks provide **interactive feedback** for architecture decisions
- Visual outputs help verify the pipeline
- No standalone script exists yet for Day 3

**Output**:
- `outputs/data_splits/` - train/val/test CSVs
- `outputs/configs/` - Model architecture, augmentation config
- `outputs/visualizations/day3_*.png` - 10 plots
- Validated training pipeline

**Time**: ~20 minutes (all 4 notebooks)

**Verdict**: ✅ Notebooks were necessary for Day 3 exploration and design.

---

## 🚀 Fast Path vs Learning Path

### **Fast Path** (Production Mode - Minimal Time)

**When to use**: You trust the pipeline and just want results.

```bash
# Step 1: Extract images (Day 1)
python src/preprocessing/convert_mat_to_png.py

# Step 2: Enhance images (Day 2)
python src/preprocessing/module1_enhance.py

# Step 3: Run Day 3 notebooks (required - no script alternative yet)
cd notebooks/day3
jupyter notebook day3_01_data_splitting.ipynb  # Run all cells
jupyter notebook day3_02_data_augmentation.ipynb  # Run all cells
jupyter notebook day3_03_cnn_architecture.ipynb  # Run all cells
jupyter notebook day3_04_training_test.ipynb  # Run all cells

# Step 4: Verify everything
python tests/day1/test_day1.py
python tests/day2/test_day2.py
python tests/day3/test_day3_completion.py
```

**Total Time**: ~25 minutes

---

### **Learning Path** (Educational Mode - Full Understanding)

**When to use**: You want to understand every step, experiment, and visualize.

```bash
# Day 1: Run ALL notebooks to understand data
cd notebooks/day1
jupyter notebook day1_dataset_explore.ipynb
jupyter notebook day1_metadata.ipynb
jupyter notebook day1_visual_check.ipynb
jupyter notebook day1_dataset_distribution_check.ipynb

# Day 2: Interactive enhancement
cd ../day2
jupyter notebook day2_enhancement.ipynb

# Day 3: Model design and testing
cd ../day3
jupyter notebook day3_01_data_splitting.ipynb
jupyter notebook day3_02_data_augmentation.ipynb
jupyter notebook day3_03_cnn_architecture.ipynb
jupyter notebook day3_04_training_test.ipynb

# Verify
cd ../..
python tests/day1/test_day1.py
python tests/day2/test_day2.py
python tests/day3/test_day3_completion.py
```

**Total Time**: ~60-90 minutes

---

## ❓ Do You NEED to Run All Notebooks?

### **Short Answer**: NO (except Day 3)

| Day | Notebooks | Are They Mandatory? | Alternative |
|-----|-----------|---------------------|-------------|
| **Day 1** | 4 notebooks | ❌ **NO** | Run `src/preprocessing/convert_mat_to_png.py` |
| **Day 2** | 1 notebook | ❌ **NO** | Run `src/preprocessing/module1_enhance.py` |
| **Day 3** | 4 notebooks | ✅ **YES** | No script alternative yet |

### **Why Day 1 & Day 2 notebooks are optional**:
- The Python scripts do the same thing (and faster)
- Notebooks add visualization and exploration
- **Use notebooks when**: Learning, debugging, presenting results
- **Use scripts when**: Production, batch processing, automation

### **Why Day 3 notebooks are required**:
- They generate critical outputs (splits, configs, architecture)
- Interactive design decisions (model architecture, augmentation params)
- Visual validation of pipeline
- **Future**: We could create a script to replace them

---

## 📝 What Each Day Actually Produces

### **Day 1 Outputs** (Required for Day 2+)
```
outputs/
├── ce_mri_images/        # ← Day 2 needs this
├── ce_mri_masks/
└── data_splits/
    └── metadata.csv      # ← Day 3 needs this
```

### **Day 2 Outputs** (Required for Day 4)
```
outputs/
└── ce_mri_enhanced/      # ← Day 4 training will use this
```

### **Day 3 Outputs** (Required for Day 4)
```
outputs/
├── data_splits/
│   ├── train_split.csv   # ← Day 4 training needs these
│   ├── val_split.csv
│   └── test_split.csv
├── configs/
│   ├── augmentation_config.json
│   └── model_architecture.json
```

---

## 🎯 Recommended Workflow

### **If starting fresh (First time)**:

```bash
# 1. Quick setup check
python tests/day1/test_day1.py  # Should fail - no outputs yet

# 2. Run Day 1 script (FAST)
python src/preprocessing/convert_mat_to_png.py
python tests/day1/test_day1.py  # Should pass now

# 3. Run Day 2 script (FAST)
python src/preprocessing/module1_enhance.py
python tests/day2/test_day2.py  # Should pass

# 4. Create metadata.csv (REQUIRED for Day 3)
# IMPORTANT: convert_mat_to_png.py does NOT create metadata.csv
# You must run: notebooks/day1/day1_metadata.ipynb
# This generates outputs/data_splits/metadata.csv

# 5. Run Day 3 notebooks (REQUIRED)
cd notebooks/day3
# Run each notebook in order:
# - day3_01_data_splitting.ipynb
# - day3_02_data_augmentation.ipynb
# - day3_03_cnn_architecture.ipynb
# - day3_04_training_test.ipynb

cd ../..
python tests/day3/test_day3_completion.py  # Should pass

# 6. Ready for Day 4!
```

**Total Time**: ~35 minutes (includes metadata creation)

---

### **If outputs already exist** (Like right now):

```bash
# Just verify everything is ready
python tests/day1/test_day1.py
python tests/day2/test_day2.py
python tests/day3/test_day3_completion.py

# All passing? You're ready for Day 4!
```

**Total Time**: ~30 seconds

---

## 🔄 When to Re-run Things

### **Re-run Day 1 if**:
- Dataset changes (new .mat files added)
- Need to regenerate images with different parameters
- Images are corrupted or missing

### **Re-run Day 2 if**:
- Want to try different enhancement parameters
- Compare original vs enhanced in ablation study
- Enhancement outputs are corrupted

### **Re-run Day 3 if**:
- Want different train/val/test splits
- Change augmentation parameters
- Redesign model architecture
- Patient leakage detected

---

## 💡 Pro Tips

### **1. Notebooks are for exploration, scripts are for production**
```python
# Notebook: "Let's try different CLAHE parameters and visualize"
# Script: "Apply the final parameters to all 3,064 images"
```

### **2. You can convert notebooks to scripts**
```bash
# Convert notebook to Python script
jupyter nbconvert --to script notebook.ipynb

# Edit and optimize for production
```

### **3. Day 4 will likely be a mix**
- Notebook for training (to see progress, plots)
- Script for final model export and evaluation

### **4. Use tests to skip unnecessary re-runs**
```bash
# Check if Day 1 outputs exist
python tests/day1/test_day1.py
# ✅ Passing? Skip Day 1, outputs are good!
```

---

## 🎬 Our Actual Execution Timeline

| Date | Phase | What We Ran | Time | Result |
|------|-------|-------------|------|--------|
| Oct 20 | Day 1 | `convert_mat_to_png.py` | 5 min | ✅ 3,064 images |
| Oct 21 | Day 2 | `module1_enhance.py` | 15 sec | ✅ 54.1% improvement |
| Oct 21 | Day 3 | All 4 notebooks | 20 min | ✅ 76.31% val accuracy |
| Oct 21 | Organize | Restructure project | 15 min | ✅ Clean directories |

**Total productive time**: ~40 minutes (excluding planning/debugging)

---

## � Can You Reproduce Everything by Deleting outputs/?

### **Short Answer: NO - Not with scripts alone!**

### **What Gets Reproduced:**

| Output Directory | Reproduced By | Can Reproduce? |
|------------------|---------------|----------------|
| `ce_mri_images/` | `convert_mat_to_png.py` | ✅ YES (5 min) |
| `ce_mri_masks/` | `convert_mat_to_png.py` | ✅ YES (5 min) |
| `ce_mri_enhanced/` | `module1_enhance.py` | ✅ YES (15 sec) |
| `data_splits/metadata.csv` | Day 1 notebooks | ⚠️ MANUAL (10 min) |
| `data_splits/*.csv` | Day 3 notebooks | ⚠️ MANUAL (20 min) |
| `configs/` | Day 3 notebooks | ⚠️ MANUAL (20 min) |
| `visualizations/` | Day 3 notebooks | ⚠️ MANUAL (20 min) |
| `training_logs/` | Day 3 notebooks | ⚠️ MANUAL (20 min) |

### **Critical Dependencies:**

1. **`metadata.csv` is NOT created by scripts!**
   - Must run `notebooks/day1/day1_metadata.ipynb`
   - Required for Day 3 data splitting
   - Without it, Day 3 notebooks will fail

2. **Day 3 outputs require notebooks**
   - No standalone scripts exist yet
   - All 4 notebooks must be run in order
   - Cannot skip for Day 4 training

### **Full Reproduction Steps:**

```bash
# Step 1: Images (scripts only - 5 min)
python src/preprocessing/convert_mat_to_png.py
python src/preprocessing/module1_enhance.py

# Step 2: Metadata (notebook required - 10 min)
jupyter notebook notebooks/day1/day1_metadata.ipynb
# OR create metadata.csv manually with columns:
# filename, label, patient_id, original_mat_name, filepath

# Step 3: Data splits + configs (notebooks required - 20 min)
jupyter notebook notebooks/day3/day3_01_data_splitting.ipynb
jupyter notebook notebooks/day3/day3_02_data_augmentation.ipynb
jupyter notebook notebooks/day3/day3_03_cnn_architecture.ipynb
jupyter notebook notebooks/day3/day3_04_training_test.ipynb

# Step 4: Verify
python tests/day3/test_day3_completion.py
```

**Total Time**: ~35 minutes (scripts + notebooks)

### **Why Not Fully Scripted?**

- Day 1 & 2 focused on exploration → scripts extracted later
- Day 3 involved design decisions → kept as notebooks
- Future: Could create scripts for Day 3 reproducibility

---

## �🚀 What's Next: Day 4

**Likely execution**:
```bash
# Option 1: Notebook (for interactive training)
jupyter notebook notebooks/day4/day4_full_training.ipynb

# Option 2: Python script (if we create one for Day 4)
python src/modeling/train_model.py --epochs 15 --patience 3

# Evaluation
jupyter notebook notebooks/day4/day4_evaluation.ipynb
```

---

## 📚 Summary

**Key Takeaways**:
1. ✅ **Day 1 & 2**: Scripts are faster, notebooks are optional
2. ✅ **Day 3**: Notebooks are currently required (no script alternative)
3. ✅ **You don't need to run everything** if outputs already exist
4. ✅ **Tests tell you what's missing** - run them first
5. ✅ **Notebooks = Learning/Exploration**, **Scripts = Production**

**Current Status**: All outputs exist, all tests passing, ready for Day 4!

---

*Last Updated: October 21, 2025*
