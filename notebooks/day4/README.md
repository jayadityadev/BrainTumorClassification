# 📘 Day 4 Notebooks - Model Training & Evaluation

**Goal:** Train CNN model on full dataset and evaluate performance

**Total Time:** 1-2 hours (mostly training time)

**Dataset:** 7,181 enhanced images from 510 patients

---

## 📚 Notebooks Overview

### **1. day4_01_full_training.ipynb** (30-60 minutes)

**Purpose:** Train CNN model with proper callbacks

**What it does:**
- ✅ Loads train/val/test splits from Day 3
- ✅ Configures data augmentation
- ✅ Builds CNN model with BatchNormalization
- ✅ Sets up callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
- ✅ Trains for up to 15 epochs (likely stops earlier)
- ✅ Saves best model automatically
- ✅ Plots training history
- ✅ Quick test set evaluation

**Model Architecture:**
```
Input (128×128×1)
  ↓
Conv Block 1 (32 filters) + BN + Dropout
  ↓
Conv Block 2 (64 filters) + BN + Dropout
  ↓
Conv Block 3 (128 filters) + BN + Dropout
  ↓
Dense (256) + BN + Dropout
Dense (128) + BN + Dropout
  ↓
Output (3 classes, Softmax)
```

**Training Configuration:**
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical Crossentropy
- **Batch Size:** 32
- **Max Epochs:** 15
- **Early Stopping:** Patience = 3 epochs

**Callbacks:**
1. **EarlyStopping:** Stops if validation loss doesn't improve for 3 epochs
2. **ModelCheckpoint:** Saves best model (lowest val_loss)
3. **ReduceLROnPlateau:** Halves learning rate if val_loss plateaus
4. **CSVLogger:** Logs training history to CSV

**Expected Results:**
- Training accuracy: 85-95%
- Validation accuracy: 75-85%
- Test accuracy: 75-85%
- Training time: 20-30 min (GPU) or 1-2 hours (CPU)

**Outputs:**
- `outputs/models/best_model_TIMESTAMP.keras`
- `outputs/training_history/training_history_TIMESTAMP.json`
- `outputs/training_history/training_history_TIMESTAMP.csv`
- `outputs/training_history/test_results_TIMESTAMP.json`
- `outputs/visualizations/day4_01_training_history_TIMESTAMP.png`

---

### **2. day4_02_model_evaluation.ipynb** (15-20 minutes)

**Purpose:** Comprehensive model evaluation on test set

**What it does:**
- ✅ Loads best trained model
- ✅ Makes predictions on 1,463 test images
- ✅ Generates confusion matrix (raw and normalized)
- ✅ Computes precision, recall, F1-score per class
- ✅ Analyzes prediction confidence
- ✅ Identifies misclassifications
- ✅ Saves complete evaluation report

**Key Metrics:**
- **Accuracy:** Overall correctness
- **Precision:** Of predicted positive, how many correct?
- **Recall:** Of actual positive, how many found?
- **F1-Score:** Harmonic mean of precision and recall

**Visualizations:**
1. Confusion matrix (counts)
2. Normalized confusion matrix (percentages)
3. Per-class metrics bar chart
4. Confidence distribution (correct vs incorrect)

**Outputs:**
- `outputs/visualizations/day4_02_confusion_matrix.png`
- `outputs/visualizations/day4_02_confusion_matrix_normalized.png`
- `outputs/visualizations/day4_02_per_class_metrics.png`
- `outputs/visualizations/day4_02_confidence_analysis.png`
- `outputs/evaluation_results/classification_report.txt`
- `outputs/evaluation_results/misclassified_samples.csv`
- `outputs/evaluation_results/evaluation_results.json`

**What to Look For:**
- **High accuracy** (>75%): Model works well!
- **Balanced per-class performance:** No single class dominates errors
- **High confidence on correct:** Model is certain when right
- **Low confidence on incorrect:** Model is uncertain when wrong (good!)

---

### **3. day4_03_predictions_analysis.ipynb** (15-20 minutes)

**Purpose:** Visualize predictions and analyze patterns

**What it does:**
- ✅ Visualizes correct predictions (high confidence)
- ✅ Visualizes correct predictions (low confidence - uncertain cases)
- ✅ Visualizes misclassifications (with detailed analysis)
- ✅ Analyzes misclassification patterns
- ✅ Shows per-class prediction examples
- ✅ Provides prediction function for new images

**Visualizations:**
1. **Correct High Confidence** (>95%): Model very certain and correct
2. **Correct Low Confidence** (<70%): Model uncertain but still correct
3. **Misclassifications**: Model wrong (sorted by confidence)
4. **Misclassification Patterns**: Which classes get confused?
5. **Per-Class Examples**: Sample predictions for each class

**Prediction Function:**
```python
results = predict_single_image('path/to/image.png', model)
# Returns: predicted_class, confidence, all_probabilities
```

**Outputs:**
- `outputs/visualizations/day4_03_correct_predictions_high_conf.png`
- `outputs/visualizations/day4_03_correct_predictions_low_conf.png`
- `outputs/visualizations/day4_03_misclassifications.png`
- `outputs/visualizations/day4_03_misclassification_patterns.png`
- `outputs/visualizations/day4_03_examples_glioma.png`
- `outputs/visualizations/day4_03_examples_meningioma.png`
- `outputs/visualizations/day4_03_examples_pituitary.png`

**Key Insights:**
- High-confidence errors = challenging cases (even for experts?)
- Low-confidence correct = model works despite uncertainty
- Misclassification patterns = systematic biases

---

## 🚀 How to Run

### **Sequential Execution (Required Order)**

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Start Jupyter
jupyter notebook notebooks/day4/

# 3. Run notebooks in order:
#    - day4_01_full_training.ipynb (WAIT for training to finish!)
#    - day4_02_model_evaluation.ipynb
#    - day4_03_predictions_analysis.ipynb
```

### **Important Notes:**

1. **Notebook 1 (Training):**
   - Takes 20-30 minutes with GPU, 1-2 hours with CPU
   - Watch progress bars - each epoch shows train/val metrics
   - Don't close notebook during training!
   - Training will likely stop early (before 15 epochs) due to EarlyStopping

2. **Notebook 2 (Evaluation):**
   - Run AFTER training completes
   - Generates comprehensive metrics
   - Review confusion matrix carefully

3. **Notebook 3 (Analysis):**
   - Run AFTER evaluation
   - Visualizes predictions
   - Use `predict_single_image()` for new images

---

## 📊 Expected Timeline

| Notebook | Time | Requires GPU? | Can Skip? |
|----------|------|---------------|-----------|
| 4.1 Training | 30-60 min | Recommended | **No** |
| 4.2 Evaluation | 15 min | No | No |
| 4.3 Analysis | 15 min | No | Optional |
| **Total** | **60-90 min** | | |

---

## 🎯 Success Criteria

After Day 4, you should have:

- [x] ✅ Trained model saved (`best_model_*.keras`)
- [x] ✅ Test accuracy > 75% (good performance)
- [x] ✅ Training curves saved (loss decreasing, accuracy increasing)
- [x] ✅ Confusion matrix generated
- [x] ✅ Classification report saved
- [x] ✅ Misclassifications identified
- [x] ✅ Prediction visualizations created
- [x] ✅ Working prediction function

---

## 🐛 Common Issues & Solutions

### Issue 1: GPU Out of Memory

**Error:** `ResourceExhaustedError: OOM when allocating tensor`

**Solution:**
```python
# In notebook 1, reduce batch size
BATCH_SIZE = 16  # Instead of 32
```

### Issue 2: Training Too Slow (CPU)

**Solution:**
- Be patient! CPU training works but is slower
- Consider using Google Colab (free GPU)
- Or reduce max epochs to 10

### Issue 3: "No module named 'tensorflow'"

**Solution:**
```bash
source .venv/bin/activate
pip install tensorflow
```

### Issue 4: Model Not Found (Notebook 2 or 3)

**Error:** `FileNotFoundError: No trained model found!`

**Solution:**
- Run notebook 1 first and wait for training to complete
- Check `outputs/models/` folder exists and contains `.keras` file

### Issue 5: Low Accuracy (<60%)

**Possible Causes:**
- Not enough epochs (training stopped too early)
- Learning rate too high/low
- Model architecture too simple

**Solutions:**
- Increase early stopping patience to 5
- Try different learning rates (0.0001 or 0.01)
- Add more convolutional layers

---

## 📈 Interpreting Results

### **Training Curves:**

**Good Training:**
```
✅ Train loss decreases smoothly
✅ Val loss decreases (slightly higher than train)
✅ Train accuracy increases
✅ Val accuracy increases (slightly lower than train)
```

**Overfitting:**
```
❌ Train loss very low, val loss high
❌ Large gap between train and val accuracy
```

**Underfitting:**
```
❌ Both train and val accuracy low
❌ Loss not decreasing
```

### **Confusion Matrix:**

**Good Model:**
- High values on diagonal (correct predictions)
- Low values off-diagonal (few confusions)

**Poor Model:**
- Many off-diagonal values
- No clear diagonal pattern

### **Test Accuracy:**

| Accuracy | Interpretation |
|----------|----------------|
| **>85%** | Excellent! 🌟 |
| **75-85%** | Good! ✅ |
| **60-75%** | Okay, could improve ⚠️ |
| **<60%** | Needs work ❌ |

---

## 💡 What You'll Learn

### **Technical Skills:**
- ✅ CNN model architecture design
- ✅ Training loop with Keras/TensorFlow
- ✅ Callback configuration
- ✅ Model evaluation metrics
- ✅ Visualization techniques
- ✅ Prediction analysis

### **Machine Learning Concepts:**
- ✅ Overfitting vs underfitting
- ✅ Early stopping strategies
- ✅ Learning rate scheduling
- ✅ Model checkpointing
- ✅ Confusion matrix interpretation
- ✅ Precision vs recall tradeoff

### **Deep Learning Best Practices:**
- ✅ BatchNormalization for stable training
- ✅ Dropout for regularization
- ✅ Data augmentation during training
- ✅ Patient-wise splitting to prevent leakage
- ✅ Proper evaluation on held-out test set

---

## 🚀 Beyond Day 4 (Optional Advanced Topics)

### **1. Transfer Learning**
Use pre-trained models (ResNet, VGG, EfficientNet):
```python
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(weights='imagenet', include_top=False)
```

### **2. Hyperparameter Tuning**
Try different:
- Learning rates: [0.0001, 0.001, 0.01]
- Batch sizes: [16, 32, 64]
- Dropout rates: [0.3, 0.5, 0.7]
- Number of filters: [16-32-64, 32-64-128, 64-128-256]

### **3. Model Ensemble**
Train multiple models and average predictions:
```python
predictions = (model1.predict(X) + model2.predict(X)) / 2
```

### **4. Explainability**
Use Grad-CAM to visualize what model looks at:
```python
from tensorflow.keras.applications.resnet50 import preprocess_input
from tf_keras_vis.gradcam import Gradcam
```

### **5. Deployment**
- Export model: `model.save('brain_tumor_classifier.keras')`
- Create Flask/FastAPI web app
- Deploy to cloud (AWS, Azure, GCP)
- Create mobile app with TensorFlow Lite

---

## 📚 References

### **Deep Learning:**
- **Keras Documentation:** https://keras.io/
- **TensorFlow Tutorials:** https://www.tensorflow.org/tutorials
- **CNN Architecture Guide:** https://cs231n.github.io/convolutional-networks/

### **Medical Imaging:**
- **Medical Image Analysis with Deep Learning:** Litjens et al., 2017
- **Brain Tumor Segmentation:** Menze et al., 2015

### **Metrics:**
- **Precision, Recall, F1-Score:** https://scikit-learn.org/stable/modules/model_evaluation.html
- **Confusion Matrix:** https://en.wikipedia.org/wiki/Confusion_matrix

---

## ✅ Checklist

Before moving to next steps:

- [ ] Training completed (notebook 1)
- [ ] Model saved to `outputs/models/`
- [ ] Training curves look good (no severe overfitting)
- [ ] Test accuracy calculated
- [ ] Confusion matrix generated (notebook 2)
- [ ] Classification report saved
- [ ] Misclassifications analyzed (notebook 3)
- [ ] Prediction function tested
- [ ] All visualizations saved

---

## 🎉 Congratulations!

If you've completed all Day 4 notebooks, you've successfully:

1. ✅ Built and trained a CNN from scratch
2. ✅ Achieved 75-85% accuracy on brain tumor classification
3. ✅ Evaluated model with proper metrics
4. ✅ Analyzed predictions and identified patterns
5. ✅ Created a working prediction pipeline

**This is a complete machine learning project!** 🎊

---

**Questions?** Check `CURRENT_STATUS.md` or `docs/PROJECT_STATUS.md`

---

*Last Updated: October 24, 2025*
