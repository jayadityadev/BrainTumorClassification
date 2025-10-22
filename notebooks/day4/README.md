# 📓 Day 4 Notebooks - Full Training & Evaluation

## 🎯 Overview

Day 4 completes the CNN training pipeline with full model training, comprehensive evaluation, ablation studies, and interpretability analysis.

---

## 📚 Notebooks

### **1️⃣ day4_01_callbacks_setup.ipynb**
**Purpose:** Configure and test training callbacks  
**Duration:** ~5 minutes  
**Key Concepts:**
- EarlyStopping (prevent overfitting)
- ModelCheckpoint (save best model)
- ReduceLROnPlateau (adaptive learning rate)
- CSVLogger (metric logging)
- Custom LearningRateTracker

**Outputs:**
- Tested callbacks with 2-epoch dry run
- Training log CSV
- LR schedule plot

---

### **2️⃣ day4_02_full_training.ipynb**
**Purpose:** Full CNN training for 10-15 epochs  
**Duration:** ~5-10 minutes (GPU)  
**Key Concepts:**
- Complete training pipeline
- Callback integration
- Training curve analysis
- Model checkpointing

**Expected Results:**
- Best validation accuracy: 78-85%
- Training time: 5-10 minutes
- Early stopping around epoch 10-12

**Outputs:**
- `model_cnn_best.h5` - Best model checkpoint
- `model_cnn_final.h5` - Final trained model
- `training_log_full.csv` - All metrics per epoch
- `training_history_full.json` - Complete history
- Training curves (accuracy & loss)
- Learning rate schedule plot
- Performance summary (JSON)

---

### **3️⃣ day4_03_evaluation.ipynb**
**Purpose:** Comprehensive test set evaluation  
**Duration:** ~10 minutes  
**Key Concepts:**
- Test set prediction
- Confusion matrix analysis
- Per-class metrics (precision, recall, F1)
- Prediction confidence
- Misclassification analysis

**Outputs:**
- Classification report (CSV)
- Confusion matrix (CSV & PNG)
- Evaluation summary (JSON)
- Sample predictions (correct & incorrect)
- Confidence distribution plots

---

### **4️⃣ day4_04_ablation_study.ipynb**
**Purpose:** Compare original vs enhanced image performance  
**Duration:** ~30 minutes (trains 2 models)  
**Key Concepts:**
- Ablation study methodology
- Baseline vs experimental comparison
- Impact quantification
- Scientific validation

**Experiment Setup:**
- **Baseline**: CNN on original images
- **Enhanced**: CNN on preprocessed images
- **Training**: 8 epochs each (reduced for speed)

**Expected Results:**
- Enhancement improvement: 5-10% validation accuracy
- Quantified impact of preprocessing module

**Outputs:**
- Ablation results table (CSV)
- Full results (JSON)
- Training curve comparison
- Bar chart comparison
- Performance summary

---

### **5️⃣ day4_05_gradcam_visualization.ipynb** (Optional)
**Purpose:** Model interpretability with Grad-CAM  
**Duration:** ~20 minutes  
**Key Concepts:**
- Gradient-weighted Class Activation Mapping
- Visual explanation of predictions
- Attention heatmaps
- Medical AI explainability

**Outputs:**
- Grad-CAM heatmaps (all samples)
- Per-class Grad-CAM visualizations
- Attention pattern analysis
- Center vs edge activation analysis

---

## 🚀 Quick Start

### **Option 1: Run All Notebooks Sequentially**

```bash
cd notebooks/day4

# 1. Setup and test callbacks
jupyter notebook day4_01_callbacks_setup.ipynb

# 2. Full training (main notebook)
jupyter notebook day4_02_full_training.ipynb

# 3. Evaluate on test set
jupyter notebook day4_03_evaluation.ipynb

# 4. Ablation study
jupyter notebook day4_04_ablation_study.ipynb

# 5. Grad-CAM (optional)
jupyter notebook day4_05_gradcam_visualization.ipynb
```

### **Option 2: Quick Training (Skip Setup)**

If you're confident with the configuration:

```bash
# Just run the main training notebook
jupyter notebook notebooks/day4/day4_02_full_training.ipynb
```

---

## 📊 Expected Timeline

| Notebook | Duration | Cumulative |
|----------|----------|------------|
| 01 - Callbacks | 5 min | 5 min |
| 02 - Training | 5-10 min | 15 min |
| 03 - Evaluation | 10 min | 25 min |
| 04 - Ablation | 30 min | 55 min |
| 05 - Grad-CAM | 20 min | 75 min |

**Total: ~1-1.5 hours** for complete Day 4

---

## 🎯 Learning Objectives

By the end of Day 4, you will understand:

### **Technical Skills:**
- ✅ How to configure training callbacks
- ✅ Full model training workflow
- ✅ Comprehensive model evaluation
- ✅ Ablation study methodology
- ✅ Model interpretability with Grad-CAM

### **Deep Learning Concepts:**
- ✅ Early stopping and overfitting prevention
- ✅ Learning rate scheduling
- ✅ Model checkpointing strategies
- ✅ Evaluation metrics for multi-class classification
- ✅ Visual explanation techniques

### **Medical AI:**
- ✅ Explainability requirements
- ✅ Trust and validation
- ✅ Regulatory considerations
- ✅ Clinical deployment readiness

---

## 📁 Output Structure

After completing Day 4, you'll have:

```
outputs/
├── models/
│   ├── model_cnn_best.h5           # Best model (highest val accuracy)
│   ├── model_cnn_final.h5          # Final model after all epochs
│   └── model_summary_final.txt     # Model architecture
├── logs/
│   ├── training_log_full.csv       # Epoch-wise metrics
│   ├── training_log_formatted.csv  # Formatted version
│   └── training_history_full.json  # Complete history
├── metrics/
│   ├── training_summary.json                # Training performance
│   ├── evaluation_summary.json              # Test set results
│   ├── classification_report.csv            # Per-class metrics
│   ├── confusion_matrix.csv                 # Confusion matrix
│   ├── per_class_metrics.csv                # Detailed metrics
│   ├── ablation_study_results.csv           # Comparison table
│   └── ablation_study_full_results.json     # Full ablation data
└── visualizations/
    ├── day4_01_lr_schedule_test.png         # LR test
    ├── day4_02_training_curves.png          # Training history
    ├── day4_02_lr_schedule.png              # Full LR schedule
    ├── day4_03_confusion_matrix.png         # Confusion matrix
    ├── day4_03_correct_predictions.png      # Correct samples
    ├── day4_03_misclassified_predictions.png # Errors
    ├── day4_03_prediction_confidence.png    # Confidence dist
    ├── day4_04_ablation_comparison.png      # Training curves
    ├── day4_04_ablation_bar_chart.png       # Bar comparison
    ├── day4_05_gradcam_all_samples.png      # All Grad-CAM
    ├── day4_05_gradcam_class_1_meningioma.png
    ├── day4_05_gradcam_class_2_glioma.png
    └── day4_05_gradcam_class_3_pituitary.png
```

---

## ⚠️ Prerequisites

Before starting Day 4, ensure:

1. ✅ **Day 3 completed** - Data splits and generators exist
2. ✅ **Enhanced images ready** - `outputs/ce_mri_enhanced/` populated
3. ✅ **Python modules** - `src/modeling/data_generator.py` and `model_cnn.py` exist
4. ✅ **GPU available** - Training will be slow on CPU (~30-60 min)

**Verify with:**
```bash
python tests/day3/test_day3.py
```

---

## 🐛 Troubleshooting

### **Issue: Out of Memory (GPU)**
```python
# Add to notebook cell:
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### **Issue: Training Too Slow**
- Reduce batch size: `BATCH_SIZE = 16`
- Reduce epochs for testing: `EPOCHS = 5`
- Use CPU if GPU unavailable (expect longer times)

### **Issue: Callbacks Not Working**
- Verify CSV log path is writable
- Check model checkpoint directory exists
- Ensure `patience` values are reasonable (2-4)

### **Issue: Grad-CAM Fails**
- Install OpenCV: `pip install opencv-python`
- Check last conv layer name is correct
- Verify model loads successfully

---

## 📈 Performance Benchmarks

### **Expected Results (GTX 1650):**

| Metric | Target | Acceptable Range |
|--------|--------|------------------|
| Training Accuracy | 85-90% | 80-95% |
| Validation Accuracy | 78-85% | 75-88% |
| Test Accuracy | 75-82% | 70-85% |
| Training Time (15 epochs) | 5-10 min | 3-15 min |
| Epochs to Best Val Acc | 8-12 | 5-15 |
| Overfitting Gap | 5-10% | <15% |

---

## 💡 Tips & Best Practices

### **For Training:**
1. **Monitor validation loss** - if it increases, model is overfitting
2. **Check LR schedule** - should reduce 1-2 times during training
3. **Early stopping is good** - don't force full 15 epochs
4. **Save both best & final** - best for deployment, final for analysis

### **For Evaluation:**
1. **Never evaluate on training data** - use test set only
2. **Check confusion matrix first** - identifies problem classes
3. **Analyze misclassifications** - look for patterns
4. **Compare confidence** - correct predictions should be more confident

### **For Ablation:**
1. **Use same architecture** - only change data
2. **Same hyperparameters** - fair comparison
3. **Reduce epochs** - 8 epochs sufficient for comparison
4. **Document findings** - quantify improvement

### **For Grad-CAM:**
1. **Check multiple samples** - not just one
2. **Validate with radiologists** - medical correctness
3. **Look for artifacts** - edges, text, markers
4. **Compare across classes** - different attention patterns?

---

## 📚 Additional Resources

### **Papers:**
- He et al., "Deep Residual Learning for Image Recognition" (ResNet)
- Selvaraju et al., "Grad-CAM: Visual Explanations" (Grad-CAM)
- Goodfellow et al., "Deep Learning Book" (Theory)

### **TensorFlow Docs:**
- [Keras Callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)
- [Model Training](https://www.tensorflow.org/guide/keras/train_and_evaluate)
- [Model Evaluation](https://www.tensorflow.org/guide/keras/evaluate_and_predict)

### **Medical AI:**
- [FDA AI/ML Guidance](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)
- [EU AI Act](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)

---

## ✅ Completion Checklist

After Day 4, you should have:

- [ ] Trained model with 78-85% validation accuracy
- [ ] Saved best model checkpoint
- [ ] Test set evaluation results
- [ ] Confusion matrix and classification report
- [ ] Ablation study comparing original vs enhanced
- [ ] Grad-CAM visualizations (optional)
- [ ] All metrics saved to CSV/JSON
- [ ] All visualizations saved to PNG
- [ ] Updated documentation

---

**Created:** October 22, 2025  
**Author:** Jayaditya Dev  
**Project:** Brain Tumor Classification  
**Status:** Ready for Execution 🚀
