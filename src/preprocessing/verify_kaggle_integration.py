"""
Verify Kaggle Dataset Integration

Quick script to verify that the Kaggle dataset integration was successful
and that the combined dataset is ready for training.

Usage:
    python src/preprocessing/verify_kaggle_integration.py

Author: Jayaditya Dev
Date: October 24, 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_success(text):
    """Print success message."""
    print(f"✅ {text}")


def print_error(text):
    """Print error message."""
    print(f"❌ {text}")


def print_info(text):
    """Print info message."""
    print(f"ℹ️  {text}")


def verify_integration():
    """Main verification function."""
    
    print_header("🔍 Kaggle Dataset Integration Verification")
    
    all_passed = True
    
    # Check 1: Combined metadata exists
    print("\n📋 Test 1: Combined Metadata File")
    combined_metadata = project_root / 'outputs' / 'data_splits' / 'metadata_combined.csv'
    
    if combined_metadata.exists():
        print_success(f"Combined metadata found: {combined_metadata}")
        df = pd.read_csv(combined_metadata)
        print_info(f"Total records: {len(df)}")
    else:
        print_error("Combined metadata not found!")
        print_info("Expected location: outputs/data_splits/metadata_combined.csv")
        print_info("Run: python src/preprocessing/integrate_kaggle_dataset.py")
        return False
    
    # Check 2: Required columns
    print("\n📋 Test 2: Metadata Structure")
    required_columns = ['filename', 'label', 'patient_id', 'original_mat_name', 'filepath', 'source']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if not missing_columns:
        print_success("All required columns present")
        print_info(f"Columns: {', '.join(df.columns)}")
    else:
        print_error(f"Missing columns: {', '.join(missing_columns)}")
        all_passed = False
    
    # Check 3: Source distribution
    print("\n📋 Test 3: Source Distribution")
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        print_success("Source distribution:")
        for source, count in source_counts.items():
            print(f"   {source}: {count} images ({count/len(df)*100:.1f}%)")
    else:
        print_error("'source' column missing")
        all_passed = False
    
    # Check 4: Class distribution
    print("\n📋 Test 4: Class Distribution")
    if 'label' in df.columns:
        class_counts = df['label'].value_counts().sort_index()
        class_names = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary'}
        
        print_success("Class distribution:")
        for label, count in class_counts.items():
            name = class_names.get(label, f'Class {label}')
            print(f"   {label} ({name}): {count} images ({count/len(df)*100:.1f}%)")
        
        # Check balance
        min_class = class_counts.min()
        max_class = class_counts.max()
        imbalance_ratio = max_class / min_class
        
        if imbalance_ratio < 3.0:
            print_success(f"Classes reasonably balanced (ratio: {imbalance_ratio:.2f}:1)")
        else:
            print_info(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}:1)")
            print_info("Consider using class weights during training")
    else:
        print_error("'label' column missing")
        all_passed = False
    
    # Check 5: Patient distribution
    print("\n📋 Test 5: Patient Distribution")
    if 'patient_id' in df.columns:
        total_patients = df['patient_id'].nunique()
        original_patients = df[df['source'] == 'original']['patient_id'].nunique() if 'source' in df.columns else 0
        kaggle_patients = df[df['source'] == 'kaggle']['patient_id'].nunique() if 'source' in df.columns else 0
        
        print_success(f"Total unique patients: {total_patients}")
        if 'source' in df.columns:
            print_info(f"Original patients: {original_patients}")
            print_info(f"Kaggle patients: {kaggle_patients}")
        
        # Images per patient
        images_per_patient = df.groupby('patient_id').size()
        print_info(f"Images per patient - Mean: {images_per_patient.mean():.1f}, "
                  f"Median: {images_per_patient.median():.1f}, "
                  f"Max: {images_per_patient.max()}")
    else:
        print_error("'patient_id' column missing")
        all_passed = False
    
    # Check 6: File existence (sample)
    print("\n📋 Test 6: File Existence (Sample Check)")
    if 'filepath' in df.columns:
        sample_size = min(100, len(df))
        sample_files = df['filepath'].sample(sample_size, random_state=42)
        
        missing_files = []
        for filepath in sample_files:
            if not Path(filepath).exists():
                missing_files.append(filepath)
        
        if not missing_files:
            print_success(f"All {sample_size} sampled files exist")
        else:
            print_error(f"{len(missing_files)}/{sample_size} sampled files not found")
            print_info("First missing file: " + str(missing_files[0]))
            all_passed = False
    else:
        print_error("'filepath' column missing")
        all_passed = False
    
    # Check 7: Data quality
    print("\n📋 Test 7: Data Quality")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print_success("No missing values detected")
    else:
        print_error("Missing values found:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"   {col}: {count} missing")
        all_passed = False
    
    # Check for duplicate filenames
    duplicate_filenames = df['filename'].duplicated().sum()
    if duplicate_filenames == 0:
        print_success("No duplicate filenames")
    else:
        print_error(f"{duplicate_filenames} duplicate filenames found")
        all_passed = False
    
    # Check 8: Kaggle images directory
    print("\n📋 Test 8: Kaggle Images Directory")
    kaggle_img_dir = project_root / 'outputs' / 'ce_mri_images_kaggle'
    
    if kaggle_img_dir.exists():
        print_success(f"Kaggle images directory exists: {kaggle_img_dir}")
        
        # Check class folders
        class_folders = [d for d in kaggle_img_dir.iterdir() if d.is_dir()]
        print_info(f"Class folders: {sorted([d.name for d in class_folders])}")
        
        # Count images
        total_kaggle_images = 0
        for class_folder in class_folders:
            images = list(class_folder.glob('*.png'))
            total_kaggle_images += len(images)
            print_info(f"   Class {class_folder.name}: {len(images)} images")
        
        print_info(f"Total Kaggle images on disk: {total_kaggle_images}")
        
        # Compare with metadata
        if 'source' in df.columns:
            metadata_kaggle_count = len(df[df['source'] == 'kaggle'])
            if total_kaggle_images == metadata_kaggle_count:
                print_success("Kaggle image count matches metadata")
            else:
                print_error(f"Mismatch: {total_kaggle_images} on disk vs {metadata_kaggle_count} in metadata")
                all_passed = False
    else:
        print_info("Kaggle images directory not found (images may be in main directory)")
    
    # Summary
    print_header("📊 Verification Summary")
    
    if all_passed:
        print_success("All verification tests passed! ✅")
        print("\n📝 Next Steps:")
        print("1. Update Day 3 notebook to use metadata_combined.csv")
        print("2. Re-run data splitting: notebooks/day3/day3_01_data_splitting.ipynb")
        print("3. Retrain model with expanded dataset")
        print("4. Compare performance: original vs combined")
    else:
        print_error("Some verification tests failed ❌")
        print("\n📝 Troubleshooting:")
        print("1. Check integration script output for errors")
        print("2. Verify Kaggle dataset was downloaded completely")
        print("3. Re-run: python src/preprocessing/integrate_kaggle_dataset.py")
    
    print("\n" + "="*70)
    
    return all_passed


if __name__ == "__main__":
    success = verify_integration()
    sys.exit(0 if success else 1)
