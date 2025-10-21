# Brain Tumor Classification Project

A deep learning project for classifying brain tumors from MRI images into three categories: Meningioma, Glioma, and Pituitary tumors.

## 📊 Dataset

- **Total Images**: 3,064 brain MRI scans
- **Classes**: 3 tumor types
  - Class 1: Meningioma (708 images)
  - Class 2: Glioma (1,426 images)
  - Class 3: Pituitary (930 images)
- **Patients**: 233 unique patients
- **Format**: Contrast-enhanced MRI images (512×512 pixels)

## 🏗️ Project Structure

```
BrainTumorProject/
├── dataset/              # Original .mat files (3,064 files)
├── outputs/
│   ├── ce_mri_images/   # Extracted PNG images
│   ├── ce_mri_masks/    # Tumor mask images
│   ├── ce_mri_enhanced/ # Enhanced images (Day 2)
│   └── visualizations/  # Analysis plots
├── src/
│   ├── convert_mat_to_png.py      # Day 1: MAT to PNG conversion
│   ├── module1_enhance.py         # Day 2: Image enhancement
│   └── visualize_enhancement.py   # Day 2: Validation plots
├── notebooks/
│   └── day2_enhancement.ipynb     # Interactive experimentation
├── docs/
│   ├── DAY1_COMPLETION_LOG.md
│   └── DAY2_COMPLETION_LOG.md
└── metadata.csv          # Image metadata with labels
```

## ✅ Completed Milestones

### Day 1: Data Extraction & Conversion ✅
- Converted 3,064 .mat files to PNG images
- Extracted tumor masks
- Generated metadata CSV with patient IDs and labels
- **Performance**: 100% success rate

### Day 2: Image Enhancement ✅
- Implemented 3-stage enhancement pipeline:
  1. **Non-Local Means Denoising** (h=10)
  2. **CLAHE** (clipLimit=2.0, tileGridSize=8×8)
  3. **Normalization** [0, 255]
- Multi-core batch processing (9 workers, ~210 images/sec)
- **Results**: 54.1% average contrast improvement
  - Class 1: +52.0%
  - Class 2: +56.9%
  - Class 3: +53.4%
- Visual validation with comparison plots

## 🚀 Upcoming Work

### Day 3: Data Augmentation & Model Preparation
- Implement data augmentation (rotations, flips, zooms)
- Patient-aware train/validation/test split (70/15/15)
- Address class imbalance

### Day 4: CNN Model Development
- Design baseline CNN architecture
- Train on enhanced images
- Evaluate performance metrics

## 🛠️ Tech Stack

- **Python**: 3.11.14
- **Libraries**: 
  - OpenCV 4.12.0 (image processing)
  - NumPy, SciPy (numerical computing)
  - Matplotlib (visualization)
  - h5py (MAT file handling)
  - scikit-image (advanced image processing)
  - tqdm (progress tracking)

## 💻 Hardware Optimization

- **CPU**: AMD Ryzen 5 5600H (12 cores @ 4.28 GHz)
- **GPU**: NVIDIA GeForce GTX 1650 Mobile
- **Optimization**: Multi-core parallel processing (75% CPU utilization)

## 📖 Documentation

Detailed completion logs available in `docs/`:
- [Day 1 Completion Log](docs/DAY1_COMPLETION_LOG.md)
- [Day 2 Completion Log](docs/DAY2_COMPLETION_LOG.md)

## 🔬 Research Context

This project focuses on automated brain tumor classification to assist medical diagnosis. The three tumor types have distinct characteristics:
- **Meningioma**: Most common, usually benign
- **Glioma**: Most aggressive, requires urgent treatment
- **Pituitary**: Affects hormone regulation

## 📝 License

This is a research/educational project.

## 👤 Author

**Jayaditya Dev**
- Email: jayadityadev261204@gmail.com

---

*Last Updated: October 21, 2025*
