"""
Update Metadata to Use Enhanced Images

This script updates the metadata_combined.csv to point to enhanced image paths
instead of raw images, enabling training with enhanced dataset.

Author: Jayaditya Dev
Date: October 24, 2025
"""

import pandas as pd
from pathlib import Path

def update_metadata_for_enhanced():
    """Update metadata to point to enhanced images."""
    
    project_root = Path(__file__).parent.parent.parent
    
    # Load existing metadata
    metadata_path = project_root / "outputs" / "data_splits" / "metadata_combined.csv"
    enhanced_metadata_path = project_root / "outputs" / "data_splits" / "metadata_enhanced.csv"
    
    print("="*70)
    print("UPDATING METADATA FOR ENHANCED IMAGES")
    print("="*70)
    
    # Read metadata
    df = pd.read_csv(metadata_path)
    print(f"\n📋 Loaded metadata: {len(df)} images")
    print(f"   Sources: {df['source'].value_counts().to_dict()}")
    
    # Create new filepath column pointing to enhanced images
    def get_enhanced_path(row):
        """Convert raw image path to enhanced image path."""
        if row['source'] == 'original':
            # outputs/ce_mri_images/1/file.png -> outputs/ce_mri_enhanced/1/file.png
            return row['filepath'].replace('ce_mri_images', 'ce_mri_enhanced')
        else:  # kaggle
            # outputs/ce_mri_images_kaggle/1/file.png -> outputs/ce_mri_enhanced_kaggle/1/file.png
            return row['filepath'].replace('ce_mri_images_kaggle', 'ce_mri_enhanced_kaggle')
    
    # Update filepaths
    df['filepath'] = df.apply(get_enhanced_path, axis=1)
    
    # Verify files exist
    print("\n🔍 Verifying enhanced images exist...")
    missing_files = []
    for idx, filepath in enumerate(df['filepath']):
        if not Path(filepath).exists():
            missing_files.append(filepath)
        if idx % 1000 == 0 and idx > 0:
            print(f"   Checked {idx}/{len(df)} files...")
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} files not found")
        print(f"   First missing: {missing_files[0]}")
    else:
        print(f"✅ All {len(df)} enhanced images exist!")
    
    # Save enhanced metadata
    df.to_csv(enhanced_metadata_path, index=False)
    
    print(f"\n✅ Enhanced metadata saved to:")
    print(f"   {enhanced_metadata_path}")
    
    print("\n📊 Summary:")
    print(f"   Total images: {len(df)}")
    print(f"   Original enhanced: {len(df[df['source']=='original'])}")
    print(f"   Kaggle enhanced: {len(df[df['source']=='kaggle'])}")
    
    print("\n📝 Next Steps:")
    print("   1. Use metadata_enhanced.csv in Day 3 notebooks")
    print("   2. Run day3_01_data_splitting.ipynb")
    print("   3. Continue with Day 3 pipeline")
    print("="*70)
    
    return enhanced_metadata_path

if __name__ == "__main__":
    update_metadata_for_enhanced()
