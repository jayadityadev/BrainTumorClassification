#!/usr/bin/env python3
"""
Module 1: Image Enhancement Pipeline
Applies Non-Local Means denoising + CLAHE to all MRI images
Optimized for multi-core CPU processing

Hardware detected:
- CPU: AMD Ryzen 5 5600H (12 cores)
- GPU: NVIDIA GeForce GTX 1650 Mobile

Usage:
    python src/module1_enhance.py
"""

import cv2
import os
import glob
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

# Paths
INPUT_ROOT = "outputs/ce_mri_images"
OUTPUT_ROOT = "outputs/ce_mri_enhanced"
LOG_DIR = "outputs/logs"

# Create output directories
os.makedirs(OUTPUT_ROOT, exist_ok=True)
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


def main():
    """Process all images in the dataset using parallel processing"""
    
    # Detect number of CPU cores
    num_cores = cpu_count()
    # Use 75% of cores to leave some for system
    num_workers = max(1, int(num_cores * 0.75))
    
    print("=" * 70)
    print("IMAGE ENHANCEMENT MODULE (Module 1)")
    print("=" * 70)
    print("\nHardware Info:")
    print(f"  CPU Cores: {num_cores} (using {num_workers} workers for parallel processing)")
    print("\nTechniques:")
    print("  1. Non-Local Means Denoising (h=10)")
    print("  2. CLAHE (clipLimit=2.0, tileGridSize=8x8)")
    print("  3. Normalization [0, 255]")
    print()
    
    error_log_path = os.path.join(LOG_DIR, "enhancement_errors.txt")
    error_count = 0
    success_count = 0
    all_errors = []
    
    # Process each class folder
    for label in ["1", "2", "3"]:
        input_dir = os.path.join(INPUT_ROOT, label)
        output_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all PNG files in this class
        img_paths = glob.glob(os.path.join(input_dir, "*.png"))
        
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
        
        # Count successes and collect errors
        for success, error_msg in results:
            if success:
                success_count += 1
            else:
                error_count += 1
                all_errors.append(error_msg)
    
    # Write errors to log file if any
    if all_errors:
        with open(error_log_path, "w") as f:
            for error in all_errors:
                f.write(f"{error}\n")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ENHANCEMENT COMPLETE")
    print("=" * 70)
    print(f"✓ Successfully processed: {success_count} images")
    if error_count > 0:
        print(f"✗ Errors: {error_count} (see {error_log_path})")
    else:
        print(f"✓ No errors")
    print(f"\nEnhanced images saved to: {OUTPUT_ROOT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
