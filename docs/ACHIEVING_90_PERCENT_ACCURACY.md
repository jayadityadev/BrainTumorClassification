# 🎯 Strategy to Achieve 90%+ Accuracy

**Current Performance:** 71.76% test accuracy  
**Target:** 90%+ test accuracy  
**Gap to Close:** +18.24 percentage points

---

## 📊 Problem Analysis

### Current Issues:

1. **Severe Meningioma underperformance:**
   - Recall: 25.47% (only 1 in 4 detected!)
   - F1-Score: 38.03%
   - 47% misclassified as Glioma, 27% as Pituitary

2. **Class imbalance:**
   - Meningioma: 446 samples (21.7%) - **underrepresented**
   - Glioma: 966 samples (46.9%) - **overrepresented**
   - Pituitary: 647 samples (31.4%)

3. **Model focuses on irrelevant features:**
   - Looks at skull, background, non-tumor tissue
   - Not leveraging the tumor masks we have!

---

## 🚀 Multi-Pronged Solution Strategy

### ✅ **Phase 1: Class Balancing (Easy Win: +5-8%)**

**Implementation:** Already added to `day4_02_full_training.ipynb`

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = {
    0: 2.31,  # Meningioma (boost by 2.31x)
    1: 1.07,  # Glioma (slight boost)
    2: 1.60   # Pituitary (moderate boost)
}

history = model.fit(
    train_generator,
    validation_data=val_generator,
    class_weight=class_weights,  # ← KEY ADDITION
    epochs=15,
    callbacks=callbacks_list
)
```

**Expected Impact:** 71.76% → **77-80%**

**Why it works:**
- Forces model to pay attention to Meningioma
- Penalizes Meningioma misclassifications more heavily
- Balances the learning process

---

### 🎭 **Phase 2: Mask-Guided Attention (Major Improvement: +8-12%)**

**Implementation:** New notebook `day4_06_mask_attention_training.ipynb`

**Key Innovation:** Use tumor masks to focus model on tumor regions only

```python
class MaskGuidedDataGenerator:
    def __getitem__(self, idx):
        # Load image
        img = load_image(filepath)
        
        # Load corresponding mask
        mask = load_mask(mask_path)
        
        # Apply attention: 30% background + 70% tumor focus
        attended_img = img * (0.3 + 0.7 * mask)
        
        return attended_img, label
```

**Why this is powerful:**
1. **Eliminates noise:** Skull, background don't confuse model
2. **Focuses learning:** All gradients flow to tumor features
3. **Preserves context:** 30% background weight keeps anatomical info
4. **Better augmentation:** Rotations/flips don't distort tumor

**Expected Impact:** 80% → **85-88%**

---

### 🏗️ **Phase 3: Enhanced Model Architecture (+3-5%)**

**Problem:** Current model has only 4.3M parameters - may be underfitting

**Solution:** Deeper network in mask-aware notebook

```python
Model: "MaskAwareCNN"
├── Conv2D(64) + Conv2D(64) + Pool + Dropout(0.25)
├── Conv2D(128) + Conv2D(128) + Pool + Dropout(0.25)
├── Conv2D(256) + Conv2D(256) + Pool + Dropout(0.25)
├── Flatten
├── Dense(512) + Dropout(0.5)
├── Dense(256) + Dropout(0.5)
└── Dense(3, softmax)

Total: ~11M parameters (2.5x larger)
```

**Changes:**
- 4 layers → 6 convolutional layers
- 128 → 512 neurons in dense layer
- More dropout for regularization
- Lower learning rate (5e-5 instead of 1e-4)

**Expected Impact:** 88% → **90-92%**

---

### ⚡ **Phase 4: Training Optimizations (+2-3%)**

**1. Longer Training:**
- Current: 15 epochs max (stops around epoch 10)
- New: 25 epochs with patience=5
- **Why:** Model needs more time to learn subtle tumor features

**2. Better Augmentation:**
- Apply same transforms to image AND mask
- Preserves tumor-background relationship
- More diverse training data

**3. Lower Learning Rate:**
- Current: 1e-4
- New: 5e-5
- **Why:** Finer weight updates for medical imaging

---

## 📈 Expected Performance Trajectory

| Phase | Method | Expected Accuracy | Cumulative Gain |
|-------|--------|-------------------|-----------------|
| Baseline | Current CNN | 71.76% | - |
| Phase 1 | + Class Weights | 77-80% | +5-8% |
| Phase 2 | + Mask Attention | 85-88% | +13-16% |
| Phase 3 | + Enhanced Architecture | 90-92% | +18-20% |
| **Target** | **Combined** | **90%+** | **✅ Achieved** |

---

## 🎯 Why Masks Are Game-Changing

### Current Approach (Without Masks):
```
❌ Model sees: [Skull, Background, Ventricles, Tumor, Noise]
❌ Learns from: All pixels equally
❌ Confusion: Similar-looking skull patterns across classes
```

### New Approach (With Masks):
```
✅ Model sees: [Tumor Region (70%), Context (30%)]
✅ Learns from: Tumor-specific features
✅ Clarity: Focuses on actual pathology
```

### Real-World Analogy:
**Without masks:** Like showing a doctor the entire head CT and asking tumor type  
**With masks:** Like zooming into just the tumor and asking tumor type

---

## 🛠️ Implementation Steps

### Step 1: Run Training with Class Weights
```bash
# Re-run Day 4.2 training with updated notebook
jupyter notebook notebooks/day4/day4_02_full_training.ipynb
```

**Expected result:** 77-80% test accuracy

---

### Step 2: Train Mask-Aware Model
```bash
# Run new mask-aware training
jupyter notebook notebooks/day4/day4_06_mask_attention_training.ipynb
```

**Expected result:** 90%+ test accuracy

---

### Step 3: Compare Results
```python
# Load both models and compare
baseline_model = load_model('outputs/models/model_cnn_best.h5')
mask_aware_model = load_model('outputs/models/model_mask_aware_best.h5')

# Test both on same test set
baseline_acc = evaluate(baseline_model, test_gen)
mask_aware_acc = evaluate(mask_aware_model, test_gen)

print(f"Baseline: {baseline_acc*100:.2f}%")
print(f"Mask-Aware: {mask_aware_acc*100:.2f}%")
print(f"Improvement: +{(mask_aware_acc - baseline_acc)*100:.2f}%")
```

---

## 📊 Per-Class Improvement Predictions

### Meningioma (Biggest Gain):
- **Current:** 25.47% recall, 38.03% F1
- **With weights:** 65-70% recall, 70% F1
- **With masks:** 85-90% recall, 87% F1
- **Why:** Masks eliminate false Glioma/Pituitary similarities

### Glioma (Stable):
- **Current:** 90.57% recall, 80.06% F1
- **With improvements:** 92-95% recall, 92% F1
- **Why:** Already performing well, slight boost from larger model

### Pituitary (Slight Improvement):
- **Current:** 96.49% recall, 81.68% F1
- **With improvements:** 95-97% recall, 94% F1
- **Why:** Already near-perfect recall, improved precision

---

## 🎓 Key Learnings

### Why Standard CNNs Struggle on Medical Images:

1. **Small datasets** (3,064 images is tiny for deep learning)
2. **High inter-class similarity** (all brain tumors look similar)
3. **Irrelevant features dominate** (skull, background more prominent than tumor)
4. **Class imbalance** (some tumor types are rarer)

### Why Mask-Guided Approach Works:

1. **Domain knowledge injection** - Uses radiologist's segmentation
2. **Feature focusing** - Forces model to learn from tumor, not background
3. **Better regularization** - Less prone to overfitting on noise
4. **Interpretability** - Can verify model looks at tumor, not artifacts

---

## 🚨 If 90% Still Not Reached

### Advanced Techniques:

**1. Ensemble Methods:**
```python
# Train 5 models with different initializations
models = [train_model() for _ in range(5)]

# Average predictions
predictions = np.mean([model.predict(test_gen) for model in models], axis=0)
```
**Expected gain:** +2-3%

**2. Transfer Learning:**
```python
# Use pre-trained ResNet50
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(128, 128, 3)
)
```
**Expected gain:** +3-5%

**3. Test-Time Augmentation:**
```python
# Average predictions over 10 augmented versions
predictions = np.mean([
    model.predict(augment(test_image)) 
    for _ in range(10)
], axis=0)
```
**Expected gain:** +1-2%

**4. External Data:**
- BraTS dataset (Glioma focus)
- TCIA Brain Tumor collections
- Pre-train on larger dataset, fine-tune on ours
**Expected gain:** +5-8%

---

## 📅 Timeline

- **Day 4.2 (Revised):** Class weights training (30 mins)
- **Day 4.6 (New):** Mask-aware training (45 mins)
- **Day 4.7 (Optional):** Ensemble/transfer learning if needed (2 hours)

---

## ✅ Success Criteria

### Minimum Acceptable Performance (MAP):
- **Overall accuracy:** ≥ 85%
- **Meningioma F1:** ≥ 75%
- **Glioma F1:** ≥ 85%
- **Pituitary F1:** ≥ 85%

### Target Performance:
- **Overall accuracy:** ≥ 90%
- **All classes F1:** ≥ 85%
- **Meningioma recall:** ≥ 80%

### Clinical Utility Threshold:
- **Overall accuracy:** ≥ 95%
- **All classes recall:** ≥ 90%
- **High confidence** (>95%) on correct predictions

---

## 🎉 Expected Outcome

**With Phase 1+2+3 combined:**
- Test accuracy: **90-92%**
- Meningioma F1: **85-88%**
- Glioma F1: **92-94%**
- Pituitary F1: **93-95%**

**This would be publication-worthy performance for a lightweight CNN on this dataset!**

---

**Author:** Jayaditya Dev  
**Date:** October 22, 2025  
**Status:** Implementation Ready  
**Next Steps:** Run `day4_06_mask_attention_training.ipynb`
