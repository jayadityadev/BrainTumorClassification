# Day 4 Advanced: Quick Reference

**Goal:** 76% → 92-95% accuracy  
**Method:** Transfer Learning + Ensemble  
**Time:** ~3-4 hours with GPU

---

## 🚀 Execution Order

```
✅ ALREADY DONE: day4_01_full_training.ipynb (76.83%)

📝 RUN THESE NOW:
1️⃣ day4_advanced_01_efficientnet.ipynb   [~60 min] → 88-92%
2️⃣ day4_advanced_02_resnet50.ipynb       [~90 min] → 89-92%
3️⃣ day4_advanced_03_densenet121.ipynb    [~70 min] → 88-91%
4️⃣ day4_advanced_04_ensemble.ipynb       [~15 min] → 92-95% 🎯
```

**IMPORTANT:** Run in order! Ensemble needs all 4 models.

---

## ⏱️ Time Budget

| Task | Time | Cumulative |
|------|------|------------|
| EfficientNetB0 | 60 min | 1h |
| ResNet50 | 90 min | 2.5h |
| DenseNet121 | 70 min | 3.7h |
| Ensemble | 15 min | 4h |

**Break Suggested:** After ResNet50 ☕

---

## 📊 Expected Results

```
Baseline CNN:     76.83% ✅
EfficientNetB0:   88-92% (target)
ResNet50:         89-92% (target)
DenseNet121:      88-91% (target)
Ensemble:         92-95% 🎯 (GOAL!)
```

---

## 🔑 Key Concepts

**Transfer Learning:**
- Use pretrained ImageNet models
- 3-phase training (freeze → partial → full)
- RGB conversion (grayscale → RGB)

**Ensemble:**
- Soft voting (average probabilities)
- Combines all 4 models
- Reduces errors, increases robustness

---

## 🛠️ What Each Notebook Does

### 1. EfficientNetB0
- Lightest model (5M params)
- Fastest training
- Excellent efficiency
- **3 phases:** 5 + 10 + 5 epochs

### 2. ResNet50
- Deep residual learning (25M params)
- Best for complex features
- Proven medical imaging performance
- **3 phases:** 5 + 12 + 8 epochs

### 3. DenseNet121
- Dense connections (8M params)
- Great gradient flow
- Feature reuse
- **3 phases:** 5 + 12 + 8 epochs

### 4. Ensemble
- Loads all 4 models
- Averages predictions
- Comprehensive evaluation
- **Inference only** (no training)

---

## ✅ Verification Checklist

Before running ensemble:
- [ ] Baseline model exists: `outputs/models/best_model_*.keras`
- [ ] EfficientNet trained: `outputs/models/transfer_learning/efficientnet_final_*.keras`
- [ ] ResNet50 trained: `outputs/models/transfer_learning/resnet50_final_*.keras`
- [ ] DenseNet121 trained: `outputs/models/transfer_learning/densenet121_final_*.keras`

Check with:
```bash
ls outputs/models/
ls outputs/models/transfer_learning/
```

---

## 🚨 Common Issues

### GPU Out of Memory
```python
# Reduce batch size in notebook
BATCH_SIZE = 16  # Instead of 32
```

### Model Not Found (Ensemble)
- Ensure all 3 transfer learning notebooks completed
- Check `outputs/models/transfer_learning/` directory
- Models saved with `*_final_*.keras` pattern

### Low Accuracy (<85%)
- Check training curves (overfitting?)
- Increase early stopping patience
- Adjust learning rates

---

## 💡 Pro Tips

1. **Run one at a time** - Prevents GPU OOM
2. **Monitor GPU** - `nvidia-smi` in terminal
3. **Check training curves** - Should be smooth
4. **Save checkpoints** - Already implemented
5. **Celebrate 92%+** - You earned it! 🎉

---

## 📈 Progress Tracking

Use this to track your runs:

```
⬜ EfficientNetB0 started: ______ (time)
⬜ EfficientNetB0 finished: ______ (accuracy: ____%)

⬜ ResNet50 started: ______ (time)
⬜ ResNet50 finished: ______ (accuracy: ____%)

⬜ DenseNet121 started: ______ (time)
⬜ DenseNet121 finished: ______ (accuracy: ____%)

⬜ Ensemble started: ______ (time)
⬜ Ensemble finished: ______ (accuracy: ____%)

🎯 Target Achieved: ⬜ Yes (92-95%) / ⬜ No
```

---

## 📁 Output Files

Each model creates:
```
outputs/
├── models/transfer_learning/
│   ├── efficientnet_final_TIMESTAMP.keras
│   ├── resnet50_final_TIMESTAMP.keras
│   └── densenet121_final_TIMESTAMP.keras
│
├── training_history/transfer_learning/
│   ├── efficientnet_history_TIMESTAMP.csv
│   ├── resnet50_history_TIMESTAMP.csv
│   └── densenet121_history_TIMESTAMP.csv
│
└── visualizations/
    ├── day4_advanced_efficientnet_history_TIMESTAMP.png
    ├── day4_advanced_resnet50_history_TIMESTAMP.png
    └── day4_advanced_densenet121_history_TIMESTAMP.png
```

Ensemble creates:
```
outputs/
├── ensemble/
│   ├── ensemble_results_TIMESTAMP.json
│   └── ensemble_predictions_TIMESTAMP.csv
│
└── visualizations/
    ├── day4_ensemble_confusion_matrix_TIMESTAMP.png
    └── day4_ensemble_comparison_TIMESTAMP.png
```

---

## 🎯 Success Indicators

### During Training:
- ✅ Val accuracy increases steadily
- ✅ Train/val gap < 10%
- ✅ Early stopping triggers (not all epochs used)
- ✅ No CUDA OOM errors

### Final Results:
- ✅ Each model > 85% accuracy
- ✅ Ensemble > 92% accuracy
- ✅ All classes > 90% accuracy
- ✅ Balanced performance

---

## 🏁 Ready? Let's Go!

1. Open `notebooks/day4/day4_advanced_01_efficientnet.ipynb`
2. Run all cells (Ctrl+Shift+Enter)
3. Wait ~60 minutes
4. Check accuracy (should be 88-92%)
5. Repeat for ResNet50 and DenseNet121
6. Run ensemble notebook
7. **Celebrate 92-95% accuracy!** 🎉

---

**For detailed guide, see:** `DAY4_ADVANCED_GUIDE.md`

**Good luck! You're about to achieve publication-quality results! 🚀**
