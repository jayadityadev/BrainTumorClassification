"""
Kaggle Brain Tumor Dataset Integration Script

This script downloads and integrates the Kaggle Brain Tumor MRI Dataset
(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
with the existing dataset while maintaining patient-wise organization.

Dataset Info:
- Source: Kaggle (masoudnickparvar/brain-tumor-mri-dataset)
- Total Images: 7,023
- Classes: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- Format: JPG images in class folders

Integration Strategy:
1. Download dataset from Kaggle
2. Filter relevant classes (exclude 'notumor')
3. Convert to grayscale PNG
4. Assign synthetic patient IDs (kaggle_0001, kaggle_0002, etc.)
5. Merge with existing metadata
6. Maintain patient-wise splitting capability

Author: Jayaditya Dev
Date: October 24, 2025
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class KaggleDatasetIntegrator:
    """Handles downloading and integrating Kaggle brain tumor dataset."""
    
    def __init__(self, project_root: str = None):
        """Initialize integrator with project paths."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.kaggle_download_dir = self.project_root / 'kaggle_temp'
        self.output_dir = self.project_root / 'outputs' / 'ce_mri_images_kaggle'
        self.existing_metadata = self.project_root / 'outputs' / 'data_splits' / 'metadata.csv'
        self.combined_metadata = self.project_root / 'outputs' / 'data_splits' / 'metadata_combined.csv'
        
        # Class mapping: Kaggle folder name -> Our label
        self.class_mapping = {
            'glioma': 2,
            'meningioma': 1,
            'pituitary': 3,
            'notumor': None  # We'll skip this class
        }
        
    def check_kaggle_setup(self):
        """Check if Kaggle API is configured."""
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        
        if not kaggle_json.exists():
            print("❌ Kaggle API not configured!")
            print("\n📝 Setup Instructions:")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Scroll to 'API' section")
            print("3. Click 'Create New Token' (downloads kaggle.json)")
            print("4. Run these commands:")
            print("   mkdir -p ~/.kaggle")
            print("   mv ~/Downloads/kaggle.json ~/.kaggle/")
            print("   chmod 600 ~/.kaggle/kaggle.json")
            print("\n5. Install kaggle: pip install kaggle")
            return False
        
        try:
            import kaggle
            print("✅ Kaggle API configured and ready!")
            return True
        except ImportError:
            print("❌ Kaggle package not installed!")
            print("Run: pip install kaggle")
            return False
    
    def download_dataset(self):
        """Download Kaggle dataset."""
        if not self.check_kaggle_setup():
            return False
        
        import kaggle
        
        print("\n📥 Downloading Kaggle Brain Tumor MRI Dataset...")
        print("   Dataset: masoudnickparvar/brain-tumor-mri-dataset")
        print(f"   Download to: {self.kaggle_download_dir}")
        
        # Create download directory
        self.kaggle_download_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download and unzip
            kaggle.api.dataset_download_files(
                'masoudnickparvar/brain-tumor-mri-dataset',
                path=str(self.kaggle_download_dir),
                unzip=True
            )
            print("✅ Dataset downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def find_dataset_structure(self):
        """Locate the actual dataset folder structure."""
        print("\n🔍 Analyzing dataset structure...")
        
        # Common possible structures
        possible_paths = [
            self.kaggle_download_dir / 'Training',
            self.kaggle_download_dir / 'Testing',
            self.kaggle_download_dir,
        ]
        
        for path in possible_paths:
            if path.exists():
                subdirs = [d for d in path.iterdir() if d.is_dir()]
                if subdirs:
                    print(f"   Found structure at: {path}")
                    print(f"   Subdirectories: {[d.name for d in subdirs]}")
                    
                    # Check if these are class folders
                    class_folders = []
                    for subdir in subdirs:
                        if subdir.name.lower() in ['glioma', 'meningioma', 'pituitary', 'notumor']:
                            class_folders.append(subdir)
                    
                    if class_folders:
                        return path, class_folders
        
        print("❌ Could not find expected dataset structure!")
        print("   Please check the downloaded files manually.")
        return None, None
    
    def process_and_integrate(self, dry_run=False):
        """Process Kaggle images and integrate with existing dataset."""
        # Find dataset
        base_path, class_folders = self.find_dataset_structure()
        if base_path is None:
            return False
        
        print(f"\n{'🔍 DRY RUN - ' if dry_run else '🔄 '}Processing Kaggle images...")
        
        # Create output directory
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track processed images
        new_metadata = []
        patient_counter = 1
        images_per_patient = 15  # Group images into synthetic patients
        
        for class_folder in class_folders:
            class_name = class_folder.name.lower()
            
            # Skip 'notumor' class
            if class_name == 'notumor':
                print(f"⏭️  Skipping '{class_name}' class")
                continue
            
            label = self.class_mapping.get(class_name)
            if label is None:
                continue
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(class_folder.glob(ext)))
            
            print(f"\n📂 Class: {class_name.upper()} (Label: {label})")
            print(f"   Found {len(image_files)} images")
            
            if dry_run:
                continue
            
            # Create output class directory
            output_class_dir = self.output_dir / str(label)
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each image
            for idx, img_file in enumerate(tqdm(image_files, desc=f"   Processing {class_name}")):
                try:
                    # Assign patient ID (group images into patients)
                    patient_id = f"kaggle_{patient_counter:04d}"
                    
                    # Load image
                    img = cv2.imread(str(img_file))
                    
                    # Convert to grayscale
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Resize to 512x512 (match original dataset)
                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                    
                    # Generate output filename
                    output_filename = f"pid{patient_id}_kaggle_{idx:04d}.png"
                    output_path = output_class_dir / output_filename
                    
                    # Save image
                    cv2.imwrite(str(output_path), img)
                    
                    # Add to metadata
                    new_metadata.append({
                        'filename': output_filename,
                        'label': label,
                        'patient_id': patient_id,
                        'original_mat_name': f'kaggle_{idx:04d}',
                        'filepath': str(output_path),
                        'source': 'kaggle'
                    })
                    
                    # Increment patient counter every N images
                    if (idx + 1) % images_per_patient == 0:
                        patient_counter += 1
                        
                except Exception as e:
                    print(f"      ⚠️  Failed to process {img_file.name}: {e}")
                    continue
            
            # Make sure to increment for the last partial group
            if len(image_files) % images_per_patient != 0:
                patient_counter += 1
        
        if dry_run:
            print("\n✅ Dry run complete - no files written")
            return True
        
        # Create metadata DataFrame
        new_df = pd.DataFrame(new_metadata)
        
        print(f"\n✅ Processed {len(new_metadata)} images from Kaggle dataset")
        print(f"   Synthetic patients created: {patient_counter - 1}")
        
        # Ensure output directory exists
        self.combined_metadata.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata if it exists
        if self.existing_metadata.exists():
            existing_df = pd.read_csv(self.existing_metadata)
            existing_df['source'] = 'original'
            
            # Combine datasets
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            print(f"\n📊 Combined Dataset Statistics:")
            print(f"   Original images: {len(existing_df)}")
            print(f"   Kaggle images: {len(new_df)}")
            print(f"   Total images: {len(combined_df)}")
            print(f"   Total patients: {combined_df['patient_id'].nunique()}")
            
            # Save combined metadata
            combined_df.to_csv(self.combined_metadata, index=False)
            print(f"\n✅ Combined metadata saved to: {self.combined_metadata}")
            
        else:
            # Only Kaggle data
            combined_df = new_df
            new_df.to_csv(self.combined_metadata, index=False)
            print(f"\n✅ Metadata saved to: {self.combined_metadata}")
        
        # Print class distribution
        print("\n📊 Class Distribution (Combined):")
        for label in sorted(combined_df['label'].unique()):
            count = len(combined_df[combined_df['label'] == label])
            percentage = count / len(combined_df) * 100
            class_names = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary'}
            print(f"   Class {label} ({class_names[label]}): {count} images ({percentage:.1f}%)")
        
        return True
    
    def cleanup(self, keep_processed=True):
        """Clean up temporary files."""
        print("\n🧹 Cleaning up...")
        
        if not keep_processed:
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
                print(f"   Removed: {self.output_dir}")
        
        if self.kaggle_download_dir.exists():
            response = input(f"   Remove download directory {self.kaggle_download_dir}? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(self.kaggle_download_dir)
                print(f"   Removed: {self.kaggle_download_dir}")
        
        print("✅ Cleanup complete")


def main():
    """Main execution function."""
    print("="*70)
    print("🧠 Kaggle Brain Tumor Dataset Integration")
    print("="*70)
    
    integrator = KaggleDatasetIntegrator()
    
    # Step 1: Download dataset
    print("\n" + "="*70)
    print("STEP 1: Download Dataset")
    print("="*70)
    
    if not integrator.kaggle_download_dir.exists() or not any(integrator.kaggle_download_dir.iterdir()):
        if not integrator.download_dataset():
            print("\n❌ Download failed. Please check Kaggle API setup.")
            return
    else:
        print(f"✅ Dataset already downloaded at: {integrator.kaggle_download_dir}")
    
    # Step 2: Dry run to preview
    print("\n" + "="*70)
    print("STEP 2: Analyze Dataset Structure (Dry Run)")
    print("="*70)
    
    integrator.process_and_integrate(dry_run=True)
    
    # Step 3: Confirm and process
    print("\n" + "="*70)
    print("STEP 3: Process and Integrate")
    print("="*70)
    
    response = input("\nProceed with integration? (y/n): ")
    if response.lower() != 'y':
        print("❌ Integration cancelled")
        return
    
    if not integrator.process_and_integrate(dry_run=False):
        print("\n❌ Integration failed")
        return
    
    # Step 4: Cleanup
    print("\n" + "="*70)
    print("STEP 4: Cleanup")
    print("="*70)
    
    integrator.cleanup(keep_processed=True)
    
    print("\n" + "="*70)
    print("✅ Integration Complete!")
    print("="*70)
    print("\n📝 Next Steps:")
    print("1. Review combined metadata: outputs/data_splits/metadata_combined.csv")
    print("2. Run data splitting notebook with combined dataset")
    print("3. Retrain model with expanded dataset")
    print("4. Compare performance: original vs combined dataset")


if __name__ == "__main__":
    main()
