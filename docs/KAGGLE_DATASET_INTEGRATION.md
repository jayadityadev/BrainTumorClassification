# 📦 Kaggle Dataset Integration Guide

## 🎯 Overview

This guide walks you through integrating the **Kaggle Brain Tumor MRI Dataset** with your existing dataset to expand training data and improve model accuracy.

**Dataset Details:**
- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Creator**: Masoud Nickparvar
- **Size**: ~7,000 images
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor (we'll exclude this)
- **Format**: JPG images in class folders

---

## 🚀 Quick Start

### Prerequisites

1. **Kaggle Account**: Create account at https://kaggle.com
2. **Kaggle API Token**: 
   - Go to https://www.kaggle.com/settings/account
   - Scroll to "API" section
   - Click "Create New Token" (downloads `kaggle.json`)

### Setup (One-Time)

```bash
# 1. Install Kaggle API
pip install kaggle

# 2. Configure Kaggle credentials
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Verify installation
kaggle datasets list
```

### Run Integration

```bash
# Activate your environment
source .venv/bin/activate

# Run the integration script
python src/preprocessing/integrate_kaggle_dataset.py
```

**The script will:**
1. ✅ Download dataset from Kaggle (~500 MB)
2. ✅ Analyze dataset structure
3. ✅ Convert images to grayscale PNG (512×512)
4. ✅ Assign synthetic patient IDs
5. ✅ Merge with existing metadata
6. ✅ Generate combined dataset statistics

**Total Time**: ~15-20 minutes

---

## 📊 What Gets Created

### Directory Structure

```
BrainTumorProject/
├── kaggle_temp/                    # Downloaded files (temp)
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── notumor/               # (excluded)
│   └── Testing/
│       └── (same structure)
│
├── outputs/
│   ├── ce_mri_images_kaggle/      # Processed Kaggle images
│   │   ├── 1/                     # Meningioma
│   │   ├── 2/                     # Glioma
│   │   └── 3/                     # Pituitary
│   │
│   └── data_splits/
│       ├── metadata.csv           # Original dataset metadata
│       └── metadata_combined.csv  # ✨ NEW: Combined metadata
```

### Metadata Structure

`metadata_combined.csv` contains:

| Column | Description | Example |
|--------|-------------|---------|
| `filename` | Image filename | `pidkaggle_0001_kaggle_0005.png` |
| `label` | Tumor class (1/2/3) | `2` |
| `patient_id` | Patient ID | `kaggle_0001` |
| `original_mat_name` | Source identifier | `kaggle_0005` |
| `filepath` | Absolute path to image | `/path/to/image.png` |
| `source` | Dataset source | `kaggle` or `original` |

---

## 📈 Expected Dataset Growth

### Before (Original Dataset)

| Class | Type | Images | Percentage |
|-------|------|--------|------------|
| 1 | Meningioma | 708 | 23.1% |
| 2 | Glioma | 1,426 | 46.5% |
| 3 | Pituitary | 930 | 30.4% |
| **Total** | | **3,064** | **100%** |

### After (Combined Dataset)

*Estimated based on Kaggle dataset distribution:*

| Class | Type | Original | Kaggle | Combined | Percentage |
|-------|------|----------|--------|----------|------------|
| 1 | Meningioma | 708 | ~1,320 | ~2,028 | ~20% |
| 2 | Glioma | 1,426 | ~3,000 | ~4,426 | ~44% |
| 3 | Pituitary | 930 | ~1,457 | ~2,387 | ~24% |
| **Total** | | **3,064** | **~5,777** | **~8,841** | **100%** |

**Growth**: ~2.9× more data! 🚀

---

## 🔄 Integration Strategy

### Patient ID Assignment

Since Kaggle images don't have real patient IDs, we create **synthetic patient IDs**:

- **Format**: `kaggle_0001`, `kaggle_0002`, etc.
- **Grouping**: ~15 images per synthetic patient
- **Purpose**: Maintain patient-wise splitting capability

### Why This Works

1. ✅ **Prevents data leakage**: Images from same "patient" stay together
2. ✅ **Maintains consistency**: Same splitting logic works
3. ✅ **Realistic distribution**: Mimics real patient scan counts
4. ✅ **Traceable**: Easy to identify source dataset

---

## 🔧 Using the Combined Dataset

### Option 1: Update Existing Notebooks

Replace metadata path in Day 3 notebooks:

```python
# OLD
meta = pd.read_csv('../../outputs/data_splits/metadata.csv')

# NEW
meta = pd.read_csv('../../outputs/data_splits/metadata_combined.csv')
```

### Option 2: Create New Training Pipeline

```bash
# Create new notebooks for combined dataset
cp notebooks/day3/day3_01_data_splitting.ipynb notebooks/day3/day3_01_data_splitting_combined.ipynb

# Edit to use metadata_combined.csv
# Run all Day 3 notebooks with combined dataset
```

### Option 3: Compare Performance

Train two models:
1. **Baseline**: Original dataset (3,064 images)
2. **Enhanced**: Combined dataset (~8,841 images)

Compare metrics:
- Training accuracy
- Validation accuracy
- Test accuracy
- Generalization (train-val gap)

---

## 📝 Next Steps After Integration

### 1. Verify Integration

```bash
# Check combined metadata
python -c "
import pandas as pd
df = pd.read_csv('outputs/data_splits/metadata_combined.csv')
print(f'Total images: {len(df)}')
print(f'Total patients: {df[\"patient_id\"].nunique()}')
print(f'Class distribution:')
print(df.groupby('label').size())
print(f'Source distribution:')
print(df.groupby('source').size())
"
```

### 2. Re-run Data Splitting

```bash
# Open Day 3 notebook 1
jupyter notebook notebooks/day3/day3_01_data_splitting.ipynb

# Update metadata path to use metadata_combined.csv
# Run all cells to generate new train/val/test splits
```

### 3. Retrain Model

```bash
# Run Day 3 notebooks with new splits
cd notebooks/day3
# Run notebooks 2-4 with combined dataset
```

### 4. Compare Results

Create comparison notebook to track:
- Original model accuracy: XX%
- Combined model accuracy: YY%
- Improvement: +ZZ%

---

## 🐛 Troubleshooting

### Issue: "Kaggle API not configured"

**Solution**:
```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/kaggle.json

# If missing, download from Kaggle settings
# Move to correct location
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: "kaggle module not found"

**Solution**:
```bash
# Activate your venv first!
source .venv/bin/activate

# Install kaggle
pip install kaggle
```

### Issue: "Dataset already downloaded but corrupted"

**Solution**:
```bash
# Remove and re-download
rm -rf kaggle_temp/
python src/preprocessing/integrate_kaggle_dataset.py
```

### Issue: "Out of memory during processing"

**Solution**:
```python
# Edit integrate_kaggle_dataset.py
# Reduce batch size or process classes one at a time
# Add memory-efficient processing
```

---

## 🔍 Quality Checks

After integration, verify:

### 1. Image Quality

```python
import cv2
import matplotlib.pyplot as plt

# Load sample Kaggle image
img_path = 'outputs/ce_mri_images_kaggle/2/pidkaggle_0001_kaggle_0001.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.title('Sample Kaggle Image (Processed)')
plt.axis('off')
plt.show()

print(f"Shape: {img.shape}")
print(f"Dtype: {img.dtype}")
print(f"Range: [{img.min()}, {img.max()}]")
```

### 2. Metadata Integrity

```python
import pandas as pd

df = pd.read_csv('outputs/data_splits/metadata_combined.csv')

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Check patient ID uniqueness
print(f"\nUnique patients: {df['patient_id'].nunique()}")
print(f"Total images: {len(df)}")

# Check file existence
missing_files = []
for filepath in df['filepath'][:100]:  # Check first 100
    if not Path(filepath).exists():
        missing_files.append(filepath)

if missing_files:
    print(f"\n⚠️  {len(missing_files)} files not found")
else:
    print("\n✅ All files exist")
```

### 3. Class Balance

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/data_splits/metadata_combined.csv')

# Class distribution
class_counts = df.groupby('label').size()
print("Class distribution:")
print(class_counts)

# Visualize
plt.figure(figsize=(10, 5))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution (Combined Dataset)')
plt.xticks([1, 2, 3], ['Meningioma', 'Glioma', 'Pituitary'])
plt.show()
```

---

## 📚 References

- **Kaggle Dataset**: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **Kaggle API Docs**: https://github.com/Kaggle/kaggle-api
- **Original Dataset**: Jun Cheng et al., "Enhanced Performance of Brain Tumor Classification"

---

## ✅ Completion Checklist

- [ ] Kaggle API configured
- [ ] Integration script executed successfully
- [ ] `metadata_combined.csv` created
- [ ] Quality checks passed
- [ ] Data splitting re-run with combined dataset
- [ ] Model retrained
- [ ] Performance comparison documented

---

*Last Updated: October 24, 2025*
*Author: Jayaditya Dev*
