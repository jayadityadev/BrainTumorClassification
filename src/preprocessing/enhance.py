#!/usr/bin/env python3
"""
Image Enhancement Pipeline
Applies Non-Local Means denoising + CLAHE to all MRI images
Works with both CE-MRI and Kaggle datasets
Optimized for multi-core CPU processing

Usage:
    python src/preprocessing/enhance.py [--dataset DATASET]
    
    --dataset: 'ce-mri', 'kaggle', or 'both' (default: 'both')
"""

import cv2
import os
import glob
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

# Paths for CE-MRI dataset
CE_MRI_INPUT = "data/ce_mri_images"
CE_MRI_OUTPUT = "data/ce_mri_enhanced"

# Paths for Kaggle dataset
KAGGLE_INPUT = "datasets/kaggle"
KAGGLE_OUTPUT = "data/kaggle_enhanced"

LOG_DIR = "outputs/logs"

# Create output directories
os.makedirs(CE_MRI_OUTPUT, exist_ok=True)
os.makedirs(KAGGLE_OUTPUT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


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
    # h=10: filter strength (good balance between noise reduction and detail preservation)
    # templateWindowSize=7: size of template patch for comparison
    # searchWindowSize=21: size of area where search is performed
    denoised = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Step 2: Apply CLAHE for adaptive contrast enhancement
    # clipLimit=2.0: threshold for contrast limiting (prevents over-amplification)
    # tileGridSize=(8,8): size of grid for local histogram equalization
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


def process_dataset(input_root, output_root, class_folders, dataset_name="Dataset"):
    """Process all images in a dataset using parallel processing"""
    
    # Detect number of CPU cores
    num_cores = cpu_count()
    # Use 75% of cores to leave some for system
    num_workers = max(1, int(num_cores * 0.75))
    
    print("=" * 70)
    print(f"IMAGE ENHANCEMENT: {dataset_name}")
    print("=" * 70)
    print("\nHardware Info:")
    print(f"  CPU Cores: {num_cores} (using {num_workers} workers for parallel processing)")
    print("\nTechniques:")
    print("  1. Non-Local Means Denoising (h=10)")
    print("  2. CLAHE (clipLimit=2.0, tileGridSize=8x8)")
    print("  3. Normalization [0, 255]")
    print()
    
    error_log_path = os.path.join(LOG_DIR, f"enhancement_errors_{dataset_name.lower().replace(' ', '_')}.txt")
    error_count = 0
    success_count = 0
    all_errors = []
    
    # Process each class folder
    for label in class_folders:
        input_dir = os.path.join(input_root, label)
        output_dir = os.path.join(output_root, label)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files in this class
        img_paths = glob.glob(os.path.join(input_dir, "*.png"))
        img_paths += glob.glob(os.path.join(input_dir, "*.jpg"))
        
        if not img_paths:
            print(f"⚠️  No images found in {input_dir}")
            continue
        
        print(f"\n📁 Processing Class {label}: {len(img_paths)} images")
        print("-" * 70)
        
        # Prepare arguments for parallel processing
        process_args = [(img_path, output_dir) for img_path in img_paths]
        
        # Process images in parallel with progress bar
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_image, process_args),
                total=len(img_paths),
                desc=f"Class {label}"
            ))
        
        # Collect results
        for success, error_msg in results:
            if success:
                success_count += 1
            else:
                error_count += 1
                all_errors.append(error_msg)
    
    # Write errors to log file if any
    if all_errors:
        with open(error_log_path, "w") as f:
            f.write("\n".join(all_errors))
    
    # Summary
    print("\n" + "=" * 70)
    print("ENHANCEMENT COMPLETE")
    print("=" * 70)
    print(f"✓ Successfully processed: {success_count} images")
    if error_count > 0:
        print(f"✗ Errors: {error_count} (see {error_log_path})")
    else:
        print(f"✓ No errors")
    print(f"\nEnhanced images saved to: {output_root}/")
    print("=" * 70)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Enhance MRI images with denoising and CLAHE')
    parser.add_argument('--dataset', type=str, default='both', 
                       choices=['ce-mri', 'kaggle', 'both'],
                       help='Which dataset to process (default: both)')
    args = parser.parse_args()
    
    if args.dataset in ['ce-mri', 'both']:
        # Process CE-MRI dataset (classes: 1, 2, 3)
        process_dataset(
            input_root=CE_MRI_INPUT,
            output_root=CE_MRI_OUTPUT,
            class_folders=["1", "2", "3"],
            dataset_name="CE-MRI Dataset"
        )
    
    if args.dataset in ['kaggle', 'both']:
        # Process Kaggle Training set
        process_dataset(
            input_root=os.path.join(KAGGLE_INPUT, "Training"),
            output_root=os.path.join(KAGGLE_OUTPUT, "Training"),
            class_folders=["glioma", "meningioma", "notumor", "pituitary"],
            dataset_name="Kaggle Training"
        )
        
        # Process Kaggle Testing set
        process_dataset(
            input_root=os.path.join(KAGGLE_INPUT, "Testing"),
            output_root=os.path.join(KAGGLE_OUTPUT, "Testing"),
            class_folders=["glioma", "meningioma", "notumor", "pituitary"],
            dataset_name="Kaggle Testing"
        )


if __name__ == "__main__":
    main()
