#!/usr/bin/env python3
"""
Directory Setup Script for Brain Tumor Classification Project
Creates all necessary directories for data, models, outputs, etc.
"""

import os
from pathlib import Path


def setup_project_directories():
    """Create all necessary directories for the project."""
    
    # Project root is parent of scripts/
    project_root = Path(__file__).parent.parent
    
    # Define all directories needed
    directories = [
        # Configuration
        "config",
        
        # Dataset directories (downloaded datasets)
        "datasets/ce-mri",           # Original .mat files from Figshare
        "datasets/kaggle",            # Kaggle dataset
        
        # Data directories (processed data)
        "data/ce_mri_images",         # Converted .mat → PNG
        "data/ce_mri_enhanced",       # Enhanced CE-MRI images
        "data/kaggle_enhanced/Training",   # Enhanced Kaggle training
        "data/kaggle_enhanced/Testing",    # Enhanced Kaggle testing
        "data/combined_dataset/train",     # Combined dataset train
        "data/combined_dataset/test",      # Combined dataset test
        "data/combined_data_splits",       # Train/test split CSVs
        "data/data_splits",                # .mat dataset splits
        
        # Model storage
        "models/current",
        "models/archive",
        "models/checkpoints",
        
        # Output directories
        "outputs/predictions",
        "outputs/logs",
        "outputs/reports",
        "outputs/visualizations",
        "outputs/training_history",
        
        # Web app uploads
        "uploads",
    ]
    
    print("🚀 Setting up project directories...\n")
    
    created_count = 0
    exists_count = 0
    
    for directory in directories:
        dir_path = project_root / directory
        if dir_path.exists():
            print(f"✓ {directory} (already exists)")
            exists_count += 1
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {directory} (created)")
            created_count += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Summary:")
    print(f"   Created: {created_count} directories")
    print(f"   Existing: {exists_count} directories")
    print(f"   Total: {created_count + exists_count} directories")
    print(f"{'='*60}")
    print("\n✅ Directory setup complete!")
    print("\n📥 Next: Download datasets")
    print("   1. CE-MRI Dataset:")
    print("      URL: https://figshare.com/ndownloader/articles/1512427/versions/5")
    print("      Extract to: datasets/ce-mri/")
    print("")
    print("   2. Kaggle Dataset:")
    print("      kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset")
    print("      Extract to: datasets/kaggle/")
    print("")
    print("🔄 Then run preprocessing:")
    print("   python src/preprocessing/convert_mat_to_png.py")
    print("   python src/preprocessing/enhance.py")
    print("")
    print("🚀 Train model:")
    print("   python src/models/fast_finetune_kaggle.py")
    print("")
    print("🌐 Launch web app:")
    print("   python app.py")


if __name__ == "__main__":
    setup_project_directories()
