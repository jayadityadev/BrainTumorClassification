#!/usr/bin/env python3
"""
Enhanced Image Processing for Combined Dataset
Applies Non-Local Means denoising + CLAHE to both original and Kaggle images

Usage:
    python src/preprocessing/enhance_combined_dataset.py
"""

import cv2
import os
import glob
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Paths
INPUT_DIRS = [
    "outputs/ce_mri_images",       # Original dataset
    "outputs/ce_mri_images_kaggle"  # Kaggle dataset
]
OUTPUT_DIRS = [
    "outputs/ce_mri_enhanced",       # Enhanced original
    "outputs/ce_mri_enhanced_kaggle"  # Enhanced Kaggle
]
LOG_DIR = "outputs/logs"

# Create output directories
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def enhance_image(img):
    """
    Apply complete enhancement pipeline:
    1. Non-Local Means Denoising
    2. CLAHE contrast enhancement
    3. Normalization
    
    Args:
        img: Input grayscale image (numpy array)
    
    Returns:
        enhanced: Enhanced image (numpy array, uint8, [0, 255])
    """
    # Step 1: Denoise using Non-Local Means
    denoised = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Step 2: Apply CLAHE for adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Step 3: Normalize to [0, 255] range
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return enhanced


def process_single_image(args):
    """
    Process a single image (for parallel processing)
    
    Args:
        args: tuple of (img_path, output_dir)
    
    Returns:
        tuple: (success: bool, error_msg: str or None)
    """
    img_path, output_dir = args
    
    try:
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return (False, f"Failed to read: {img_path}")
        
        # Apply enhancement
        enhanced = enhance_image(img)
        
        # Save enhanced image with same filename
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, enhanced)
        
        return (True, None)
        
    except Exception as e:
        return (False, f"{img_path}\t{repr(e)}")


def process_dataset(input_root, output_root, dataset_name, num_workers):
    """Process all images in a dataset"""
    
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*70}")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}\n")
    
    error_count = 0
    success_count = 0
    all_errors = []
    
    # Process each class folder
    for label in ["1", "2", "3"]:
        input_dir = os.path.join(input_root, label)
        output_dir = os.path.join(output_root, label)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all PNG files in this class
        img_paths = glob.glob(os.path.join(input_dir, "*.png"))
        
        if not img_paths:
            print(f"⚠️  No images found in {input_dir}")
            continue
        
        class_names = {'1': 'Meningioma', '2': 'Glioma', '3': 'Pituitary'}
        print(f"📁 Class {label} ({class_names[label]}): {len(img_paths)} images")
        
        # Prepare arguments for parallel processing
        process_args = [(img_path, output_dir) for img_path in img_paths]
        
        # Process images in parallel with progress bar
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_image, process_args),
                total=len(img_paths),
                desc=f"   Enhancing Class {label}"
            ))
        
        # Count successes and collect errors
        for success, error_msg in results:
            if success:
                success_count += 1
            else:
                error_count += 1
                all_errors.append(error_msg)
    
    return success_count, error_count, all_errors


def main():
    """Process all images in both datasets using parallel processing"""
    
    # Detect number of CPU cores
    num_cores = cpu_count()
    num_workers = max(1, int(num_cores * 0.75))
    
    print("="*70)
    print("COMBINED DATASET ENHANCEMENT")
    print("="*70)
    print("\n🔧 Hardware Info:")
    print(f"   CPU Cores: {num_cores} (using {num_workers} workers)")
    print("\n🎨 Enhancement Pipeline:")
    print("   1. Non-Local Means Denoising (h=10)")
    print("   2. CLAHE (clipLimit=2.0, tileGridSize=8x8)")
    print("   3. Normalization [0, 255]")
    
    total_success = 0
    total_errors = 0
    all_errors = []
    
    # Process each dataset
    dataset_names = ["Original Dataset", "Kaggle Dataset"]
    
    for input_dir, output_dir, name in zip(INPUT_DIRS, OUTPUT_DIRS, dataset_names):
        if not Path(input_dir).exists():
            print(f"\n⚠️  Skipping {name}: {input_dir} not found")
            continue
        
        success, errors, error_msgs = process_dataset(input_dir, output_dir, name, num_workers)
        total_success += success
        total_errors += errors
        all_errors.extend(error_msgs)
    
    # Write errors to log file if any
    if all_errors:
        error_log_path = os.path.join(LOG_DIR, "enhancement_errors_combined.txt")
        with open(error_log_path, "w") as f:
            for error in all_errors:
                f.write(f"{error}\n")
        print(f"\n⚠️  Error log: {error_log_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("✅ ENHANCEMENT COMPLETE")
    print("="*70)
    print(f"✓ Successfully processed: {total_success:,} images")
    if total_errors > 0:
        print(f"✗ Errors: {total_errors}")
    else:
        print(f"✓ No errors - All images enhanced successfully!")
    
    print("\n📂 Enhanced images saved to:")
    for output_dir in OUTPUT_DIRS:
        if Path(output_dir).exists():
            print(f"   - {output_dir}/")
    
    print("\n📝 Next Steps:")
    print("   1. Verify enhanced images")
    print("   2. Update metadata to use enhanced image paths")
    print("   3. Run Day 3 notebooks for data splitting")
    print("="*70)


if __name__ == "__main__":
    main()
