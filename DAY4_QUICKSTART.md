# 🚀 Day 4 Quick Start Guide

**Ready to train your brain tumor classifier? Follow these simple steps!**

---

## ✅ Prerequisites Check

Before starting, make sure:

```bash
# 1. You have train/val/test splits
ls outputs/data_splits/*_split.csv

# Expected:
# - train_split.csv (~4,863 images)
# - val_split.csv (~855 images)
# - test_split.csv (~1,463 images)

# 2. Virtual environment is activated
source .venv/bin/activate

# 3. Check TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

If any checks fail, see [Troubleshooting](#troubleshooting) below.

---

## 🎯 Three Simple Steps

### **Step 1: Train Model** (30-60 min)

```bash
jupyter notebook notebooks/day4/day4_01_full_training.ipynb
```

**What happens:**
- Model trains for up to 15 epochs (will likely stop early)
- You'll see progress bars for each epoch
- Best model saves automatically
- Expected accuracy: 75-85%

**Don't worry if:**
- Training stops before 15 epochs (that's early stopping working!)
- Training accuracy is higher than validation (that's normal)
- Each epoch takes 1-2 minutes on GPU or 5-10 minutes on CPU

☕ **Grab a coffee! This takes time.**

---

### **Step 2: Evaluate Model** (15 min)

```bash
jupyter notebook notebooks/day4/day4_02_model_evaluation.ipynb
```

**What happens:**
- Loads your trained model
- Tests on 1,463 test images
- Generates confusion matrix
- Computes precision/recall/F1-score

**You'll get:**
- Overall test accuracy
- Per-class performance metrics
- Confusion matrix visualization
- Misclassification analysis

---

### **Step 3: Analyze Predictions** (15 min)

```bash
jupyter notebook notebooks/day4/day4_03_predictions_analysis.ipynb
```

**What happens:**
- Visualizes correct predictions
- Shows misclassifications
- Analyzes patterns
- Creates prediction function for new images

**You'll see:**
- High confidence correct predictions
- Uncertain but correct predictions
- Misclassifications with explanations
- Per-class examples

---

## 📊 What to Expect

### **Training Output (Step 1):**

```
Epoch 1/15
152/152 [==============================] - 45s 295ms/step
loss: 0.8234 - accuracy: 0.6512 - val_loss: 0.5234 - val_accuracy: 0.7891

Epoch 2/15
152/152 [==============================] - 42s 276ms/step
loss: 0.4567 - accuracy: 0.8234 - val_loss: 0.4123 - val_accuracy: 0.8345

...

✅ Training completed!
Test Accuracy: 82.45%
```

### **Evaluation Output (Step 2):**

```
📊 TEST SET ACCURACY: 0.8245 (82.45%)

Per-Class Performance:
  glioma:      Precision: 0.8123  Recall: 0.8456  F1: 0.8287
  meningioma:  Precision: 0.7891  Recall: 0.7654  F1: 0.7771
  pituitary:   Precision: 0.8678  Recall: 0.8890  F1: 0.8783
```

### **Files Created:**

```
outputs/
├── models/
│   └── best_model_20251024_193045.keras       ← Your trained model!
├── training_history/
│   ├── training_history_20251024_193045.json
│   ├── training_history_20251024_193045.csv
│   └── test_results_20251024_193045.json
├── evaluation_results/
│   ├── classification_report.txt
│   ├── misclassified_samples.csv
│   └── evaluation_results.json
└── visualizations/
    ├── day4_01_training_history_*.png
    ├── day4_02_confusion_matrix.png
    ├── day4_02_per_class_metrics.png
    ├── day4_03_correct_predictions_high_conf.png
    ├── day4_03_misclassifications.png
    └── ... (15+ visualizations)
```

---

## 💡 Tips for Success

### **During Training (Step 1):**

1. **Watch the metrics:**
   - Loss should decrease
   - Accuracy should increase
   - Val_accuracy should be close to train accuracy (within 5-10%)

2. **Don't panic if:**
   - Training stops at epoch 8-12 (early stopping is working!)
   - Val_accuracy fluctuates slightly (normal)
   - Progress bars seem slow on CPU (expected)

3. **Know when to worry:**
   - ❌ Val_accuracy much lower than train_accuracy (>20% gap) = overfitting
   - ❌ Both accuracies stuck at 33% = model not learning (random guessing)
   - ❌ Loss increasing = something wrong

### **During Evaluation (Step 2):**

1. **Good signs:**
   - ✅ Test accuracy close to validation accuracy
   - ✅ Similar performance across all classes
   - ✅ High diagonal values in confusion matrix

2. **Red flags:**
   - ❌ Test accuracy much lower than validation = model doesn't generalize
   - ❌ One class has 0% recall = model never predicts that class
   - ❌ Confusion matrix shows all predictions go to one class

### **During Analysis (Step 3):**

1. **Interesting to check:**
   - High-confidence misclassifications (model very wrong!)
   - Low-confidence correct predictions (model got lucky?)
   - Which classes get confused most

2. **Use for improvement:**
   - See if misclassified images are genuinely hard
   - Check if certain patients cause more errors
   - Identify patterns in mistakes

---

## 🐛 Troubleshooting

### **Problem:** "No module named 'tensorflow'"

```bash
source .venv/bin/activate
pip install tensorflow
```

### **Problem:** "GPU out of memory"

In notebook 1, change:
```python
BATCH_SIZE = 16  # Reduce from 32
```

### **Problem:** "No trained model found" (in notebook 2 or 3)

Run notebook 1 first and **wait for training to finish completely**.

### **Problem:** Training stuck at low accuracy (~33%)

Try:
- Increase max epochs to 20
- Change learning rate to 0.0001
- Check if data generators are working (run first few cells)

### **Problem:** Very slow training (CPU)

Options:
- Be patient (it will work, just slower)
- Use Google Colab (free GPU): https://colab.research.google.com
- Reduce epochs to 10

---

## 🎯 Success Criteria

You're done when:

- [x] ✅ Training completed without errors
- [x] ✅ Best model saved in `outputs/models/`
- [x] ✅ Test accuracy > 70% (ideally > 75%)
- [x] ✅ Confusion matrix generated
- [x] ✅ Classification report saved
- [x] ✅ Visualizations created

---

## 📞 Quick Reference

### **Commands:**

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter
jupyter notebook notebooks/day4/

# Check GPU (optional)
nvidia-smi

# View model files
ls -lh outputs/models/

# View training history
cat outputs/training_history/*.json | grep accuracy
```

### **Important Files:**

- **Training notebook:** `notebooks/day4/day4_01_full_training.ipynb`
- **Evaluation notebook:** `notebooks/day4/day4_02_model_evaluation.ipynb`
- **Analysis notebook:** `notebooks/day4/day4_03_predictions_analysis.ipynb`
- **Best model:** `outputs/models/best_model_*.keras`
- **Results:** `outputs/evaluation_results/`

---

## 🎉 What's Next?

After completing Day 4:

1. **Review Results:**
   - Check confusion matrix
   - Read classification report
   - Look at misclassified samples

2. **Optional Improvements:**
   - Try transfer learning (ResNet, VGG)
   - Tune hyperparameters
   - Add more data augmentation

3. **Deploy Model:**
   - Create simple web app
   - Export for mobile (TensorFlow Lite)
   - Share your results!

---

## 💬 Need Help?

- **Detailed Guide:** See `notebooks/day4/README.md`
- **Project Status:** See `CURRENT_STATUS.md`
- **Full Documentation:** See `docs/PROJECT_STATUS.md`

---

**Ready? Let's train that model!** 🚀

```bash
source .venv/bin/activate
jupyter notebook notebooks/day4/day4_01_full_training.ipynb
```

---

*Good luck! You've got this!* 💪
