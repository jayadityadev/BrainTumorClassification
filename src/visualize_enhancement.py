#!/usr/bin/env python3
"""
Visual Validation: Compare Original vs Enhanced Images
Creates side-by-side comparison plots
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import glob
import os
import random

# Paths
ORIG_DIR = "outputs/ce_mri_images"
ENH_DIR = "outputs/ce_mri_enhanced"
VIS_DIR = "outputs/visualizations"

os.makedirs(VIS_DIR, exist_ok=True)

print("=" * 70)
print("VISUAL VALIDATION: Original vs Enhanced")
print("=" * 70)

# Create comparison for 3 random samples from each class
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i, label in enumerate(['1', '2', '3']):
    # Pick a random image from this class
    orig_files = glob.glob(os.path.join(ORIG_DIR, label, "*.png"))
    if not orig_files:
        continue
    
    sample_file = random.choice(orig_files)
    filename = os.path.basename(sample_file)
    
    # Load original and enhanced
    img_orig = cv2.imread(sample_file, cv2.IMREAD_GRAYSCALE)
    img_enh = cv2.imread(os.path.join(ENH_DIR, label, filename), cv2.IMREAD_GRAYSCALE)
    
    # Compute metrics
    orig_std = img_orig.std()
    enh_std = img_enh.std()
    contrast_improvement = ((enh_std / orig_std - 1) * 100)
    
    # Class labels
    class_names = {'1': 'Meningioma', '2': 'Glioma', '3': 'Pituitary'}
    
    # Plot original
    axes[i, 0].imshow(img_orig, cmap='gray')
    axes[i, 0].set_title(f'Class {label} ({class_names[label]})\nOriginal', 
                         fontsize=10, fontweight='bold')
    axes[i, 0].axis('off')
    
    # Plot enhanced
    axes[i, 1].imshow(img_enh, cmap='gray')
    axes[i, 1].set_title(f'Enhanced\n(+{contrast_improvement:.1f}% contrast)', 
                         fontsize=10, fontweight='bold', color='green')
    axes[i, 1].axis('off')
    
    # Plot histograms
    axes[i, 2].hist(img_orig.ravel(), bins=256, range=(0, 256), 
                    color='blue', alpha=0.7, label='Original')
    axes[i, 2].set_title('Histogram: Original', fontsize=10)
    axes[i, 2].set_xlim([0, 255])
    axes[i, 2].grid(alpha=0.3)
    
    axes[i, 3].hist(img_enh.ravel(), bins=256, range=(0, 256), 
                    color='green', alpha=0.7, label='Enhanced')
    axes[i, 3].set_title('Histogram: Enhanced', fontsize=10)
    axes[i, 3].set_xlim([0, 255])
    axes[i, 3].grid(alpha=0.3)
    
    print(f"✓ Class {label} - Contrast improvement: +{contrast_improvement:.1f}%")

plt.tight_layout()
save_path = os.path.join(VIS_DIR, "enhancement_comparison.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved comparison plot: {save_path}")

# Create a detailed comparison grid (6 samples)
fig, axes = plt.subplots(6, 2, figsize=(10, 18))

samples_per_class = 2
idx = 0

for label in ['1', '2', '3']:
    orig_files = glob.glob(os.path.join(ORIG_DIR, label, "*.png"))
    samples = random.sample(orig_files, min(samples_per_class, len(orig_files)))
    
    for sample_file in samples:
        filename = os.path.basename(sample_file)
        
        img_orig = cv2.imread(sample_file, cv2.IMREAD_GRAYSCALE)
        img_enh = cv2.imread(os.path.join(ENH_DIR, label, filename), cv2.IMREAD_GRAYSCALE)
        
        # Original
        axes[idx, 0].imshow(img_orig, cmap='gray')
        axes[idx, 0].set_title(f'Class {label}: Original\n{filename[:30]}...', fontsize=9)
        axes[idx, 0].axis('off')
        
        # Enhanced
        axes[idx, 1].imshow(img_enh, cmap='gray')
        axes[idx, 1].set_title(f'Class {label}: Enhanced', fontsize=9, color='green')
        axes[idx, 1].axis('off')
        
        idx += 1

plt.tight_layout()
grid_path = os.path.join(VIS_DIR, "enhancement_grid.png")
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved detailed grid: {grid_path}")

print("\n" + "=" * 70)
print("VISUAL VALIDATION COMPLETE")
print("=" * 70)
print(f"Generated visualizations:")
print(f"  - {save_path}")
print(f"  - {grid_path}")
print("=" * 70)
