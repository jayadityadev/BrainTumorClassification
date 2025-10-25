# 🚀 Kaggle Dataset Integration - Quick Start

## ⚡ 5-Minute Setup

### Step 1: Install Kaggle API (30 seconds)

```bash
# Activate environment
source .venv/bin/activate

# Install kaggle
pip install kaggle
```

### Step 2: Configure Kaggle Credentials (2 minutes)

1. Go to https://www.kaggle.com/settings/account
2. Scroll to **API** section
3. Click **"Create New Token"** → Downloads `kaggle.json`
4. Run these commands:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Run Integration Script (15-20 minutes)

```bash
cd /projects/ai-ml/BrainTumorProject
python src/preprocessing/integrate_kaggle_dataset.py
```

**The script will:**
- ✅ Download ~500 MB from Kaggle
- ✅ Process ~5,777 images
- ✅ Create combined metadata
- ✅ Show statistics

**Just follow the prompts!**

---

## 📊 What You'll Get

### Before
- 3,064 images
- 233 patients
- 3 classes

### After
- **~8,841 images** (2.9× more!)
- **~618 patients** (2.7× more!)
- 3 classes (same)

---

## 🔄 Next Steps After Integration

### Quick Path (10 minutes)

```bash
# 1. Verify integration
python -c "import pandas as pd; df=pd.read_csv('outputs/data_splits/metadata_combined.csv'); print(f'Total: {len(df)} images, {df[\"patient_id\"].nunique()} patients')"

# 2. Open Day 3 notebook 1
jupyter notebook notebooks/day3/day3_01_data_splitting.ipynb

# 3. Change this line:
#    meta = pd.read_csv('../../outputs/data_splits/metadata.csv')
# To:
#    meta = pd.read_csv('../../outputs/data_splits/metadata_combined.csv')

# 4. Run all cells → new train/val/test splits

# 5. Run Day 3 notebooks 2-4 with new splits

# 6. Train model → Compare results!
```

---

## 🎯 Expected Performance Improvement

With 2.9× more training data, you should see:
- ✅ Higher accuracy (+5-15%)
- ✅ Better generalization (smaller train-val gap)
- ✅ More robust model
- ✅ Reduced overfitting

---

## 📚 Full Documentation

See `docs/KAGGLE_DATASET_INTEGRATION.md` for:
- Detailed troubleshooting
- Quality checks
- Performance comparison guide
- Advanced configurations

---

## 🆘 Quick Troubleshooting

**❌ "Kaggle API not configured"**
→ Follow Step 2 above

**❌ "kaggle module not found"**
→ Run: `pip install kaggle`

**❌ "Permission denied: kaggle.json"**
→ Run: `chmod 600 ~/.kaggle/kaggle.json`

**❌ "Dataset already exists"**
→ Delete `kaggle_temp/` folder and re-run

---

## ✅ Ready to Go!

You're all set! Run the integration script and watch your dataset grow 🚀

```bash
python src/preprocessing/integrate_kaggle_dataset.py
```

*Questions? Check `docs/KAGGLE_DATASET_INTEGRATION.md`*
