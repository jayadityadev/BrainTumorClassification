"""
Combine .mat dataset and Kaggle dataset for unified training
This will create a new combined dataset with both sources
"""

import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2

# Paths
MAT_ENHANCED_DIR = 'data/ce_mri_enhanced'
KAGGLE_TRAIN_DIR = 'data/kaggle_enhanced/Training'
KAGGLE_TEST_DIR = 'data/kaggle_enhanced/Testing'
OUTPUT_DIR = 'data/combined_dataset'
OUTPUT_SPLITS_DIR = 'data/combined_data_splits'

# Class mappings
CLASS_TO_LABEL = {'glioma': 0, 'meningioma': 1, 'pituitary': 2}
LABEL_TO_CLASS = {0: 'glioma', 1: 'meningioma', 2: 'pituitary'}

# For .mat dataset (uses different label encoding)
MAT_LABEL_TO_CLASS = {1: 'meningioma', 2: 'glioma', 3: 'pituitary'}
MAT_LABEL_TO_NEW_LABEL = {1: 1, 2: 0, 3: 2}  # Convert to standard 0, 1, 2

def preprocess_image_to_rgb(img_path, target_size=(128, 128)):
    """Load and preprocess image to RGB format with center crop"""
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)
    
    # Center crop
    h, w = img_array.shape
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    img_cropped = img_array[start_h:start_h+min_dim, start_w:start_w+min_dim]
    
    # Resize
    img_resized = Image.fromarray(img_cropped).resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    
    # Convert to RGB
    img_rgb = np.stack([img_array] * 3, axis=-1)
    
    return Image.fromarray(img_rgb.astype(np.uint8))


def create_mat_splits(enhanced_dir, splits_dir, train_ratio=0.8):
    """Create train/test splits from .mat enhanced images"""
    import glob
    from sklearn.model_selection import train_test_split
    
    os.makedirs(splits_dir, exist_ok=True)
    
    all_data = []
    
    # Collect all enhanced images
    for label in ['1', '2', '3']:
        class_dir = os.path.join(enhanced_dir, label)
        if not os.path.exists(class_dir):
            continue
        
        img_files = glob.glob(os.path.join(class_dir, '*.png'))
        
        for img_path in img_files:
            all_data.append({
                'filepath': img_path,
                'label': int(label),
                'class_name': MAT_LABEL_TO_CLASS[int(label)]
            })
    
    # Split into train/test
    train_data, test_data = train_test_split(
        all_data,
        train_size=train_ratio,
        random_state=42,
        stratify=[d['label'] for d in all_data]
    )
    
    # Save as CSVs
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_df.to_csv(os.path.join(splits_dir, 'train_split.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test_split.csv'), index=False)
    
    print(f"   ✓ Created splits: {len(train_df)} train, {len(test_df)} test")


def combine_datasets():
    """Combine .mat and Kaggle datasets"""
    
    print("="*80)
    print("🔄 COMBINING DATASETS")
    print("="*80)
    print()
    
    # Create output directories
    for split in ['train', 'test']:
        for class_name in ['glioma', 'meningioma', 'pituitary']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)
    
    os.makedirs(OUTPUT_SPLITS_DIR, exist_ok=True)
    
    all_data = []
    image_counter = 0
    
    # ============================================================
    # 1. Process .mat dataset (from existing splits)
    # ============================================================
    print("📂 Processing .mat dataset...")
    
    # Check if we have existing splits, otherwise we'll need to create them
    mat_splits_dir = 'data/data_splits'
    train_split_path = os.path.join(mat_splits_dir, 'train_split.csv')
    test_split_path = os.path.join(mat_splits_dir, 'test_split.csv')
    
    if not os.path.exists(train_split_path):
        print("   ⚠️  No existing .mat dataset splits found.")
        print("   Creating new splits from enhanced images...")
        create_mat_splits(MAT_ENHANCED_DIR, mat_splits_dir)
    
    # Load existing splits
    mat_train_df = pd.read_csv(train_split_path)
    mat_test_df = pd.read_csv(test_split_path)
    
    print(f"   .mat training images: {len(mat_train_df)}")
    print(f"   .mat testing images: {len(mat_test_df)}")
    
    # Process training split
    for idx, row in tqdm(mat_train_df.iterrows(), total=len(mat_train_df), desc="  Processing .mat train"):
        src_path = row['filepath']
        if not os.path.exists(src_path):
            continue
        
        # Convert .mat label to standard class name
        mat_label = row['label']
        class_name = MAT_LABEL_TO_CLASS[mat_label]
        new_label = MAT_LABEL_TO_NEW_LABEL[mat_label]
        
        # New filename
        new_filename = f"mat_{image_counter:06d}.png"
        dst_path = os.path.join(OUTPUT_DIR, 'train', class_name, new_filename)
        
        # Preprocess to RGB and save
        try:
            img = preprocess_image_to_rgb(src_path)
            img.save(dst_path)
            
            all_data.append({
                'filepath': dst_path,
                'filename': new_filename,
                'label': new_label,
                'class_name': class_name,
                'split': 'train',
                'source': 'mat'
            })
            image_counter += 1
        except Exception as e:
            print(f"    Error processing {src_path}: {e}")
    
    # Process testing split
    for idx, row in tqdm(mat_test_df.iterrows(), total=len(mat_test_df), desc="  Processing .mat test"):
        src_path = row['filepath']
        if not os.path.exists(src_path):
            continue
        
        # Convert .mat label to standard class name
        mat_label = row['label']
        class_name = MAT_LABEL_TO_CLASS[mat_label]
        new_label = MAT_LABEL_TO_NEW_LABEL[mat_label]
        
        # New filename
        new_filename = f"mat_{image_counter:06d}.png"
        dst_path = os.path.join(OUTPUT_DIR, 'test', class_name, new_filename)
        
        # Preprocess to RGB and save
        try:
            img = preprocess_image_to_rgb(src_path)
            img.save(dst_path)
            
            all_data.append({
                'filepath': dst_path,
                'filename': new_filename,
                'label': new_label,
                'class_name': class_name,
                'split': 'test',
                'source': 'mat'
            })
            image_counter += 1
        except Exception as e:
            print(f"    Error processing {src_path}: {e}")
    
    print()
    
    # ============================================================
    # 2. Process Kaggle dataset
    # ============================================================
    print("📂 Processing Kaggle dataset...")
    
    # Process Kaggle training
    kaggle_train_count = 0
    for class_name in ['glioma', 'meningioma', 'pituitary']:
        class_dir = os.path.join(KAGGLE_TRAIN_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
        
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc=f"  Processing Kaggle train/{class_name}"):
            src_path = os.path.join(class_dir, img_file)
            
            # New filename
            new_filename = f"kaggle_{image_counter:06d}.png"
            dst_path = os.path.join(OUTPUT_DIR, 'train', class_name, new_filename)
            
            # Preprocess to RGB and save
            try:
                img = preprocess_image_to_rgb(src_path)
                img.save(dst_path)
                
                all_data.append({
                    'filepath': dst_path,
                    'filename': new_filename,
                    'label': CLASS_TO_LABEL[class_name],
                    'class_name': class_name,
                    'split': 'train',
                    'source': 'kaggle'
                })
                image_counter += 1
                kaggle_train_count += 1
            except Exception as e:
                print(f"    Error processing {src_path}: {e}")
    
    print(f"   Kaggle training images: {kaggle_train_count}")
    
    # Process Kaggle testing
    kaggle_test_count = 0
    for class_name in ['glioma', 'meningioma', 'pituitary']:
        class_dir = os.path.join(KAGGLE_TEST_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
        
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc=f"  Processing Kaggle test/{class_name}"):
            src_path = os.path.join(class_dir, img_file)
            
            # New filename
            new_filename = f"kaggle_{image_counter:06d}.png"
            dst_path = os.path.join(OUTPUT_DIR, 'test', class_name, new_filename)
            
            # Preprocess to RGB and save
            try:
                img = preprocess_image_to_rgb(src_path)
                img.save(dst_path)
                
                all_data.append({
                    'filepath': dst_path,
                    'filename': new_filename,
                    'label': CLASS_TO_LABEL[class_name],
                    'class_name': class_name,
                    'split': 'test',
                    'source': 'kaggle'
                })
                image_counter += 1
                kaggle_test_count += 1
            except Exception as e:
                print(f"    Error processing {src_path}: {e}")
    
    print(f"   Kaggle testing images: {kaggle_test_count}")
    print()
    
    # ============================================================
    # 3. Create DataFrame and save splits
    # ============================================================
    df = pd.DataFrame(all_data)
    
    # Split into train/test
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # Save splits
    train_csv_path = os.path.join(OUTPUT_SPLITS_DIR, 'train_split.csv')
    test_csv_path = os.path.join(OUTPUT_SPLITS_DIR, 'test_split.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    # ============================================================
    # 4. Print statistics
    # ============================================================
    print("="*80)
    print("📊 COMBINED DATASET STATISTICS")
    print("="*80)
    print()
    
    print("📂 Training Set:")
    print(f"   Total: {len(train_df)} images")
    for class_name in ['glioma', 'meningioma', 'pituitary']:
        class_count = len(train_df[train_df['class_name'] == class_name])
        mat_count = len(train_df[(train_df['class_name'] == class_name) & (train_df['source'] == 'mat')])
        kaggle_count = len(train_df[(train_df['class_name'] == class_name) & (train_df['source'] == 'kaggle')])
        print(f"   {class_name.capitalize()}: {class_count} (.mat: {mat_count}, kaggle: {kaggle_count})")
    
    print()
    print("📂 Testing Set:")
    print(f"   Total: {len(test_df)} images")
    for class_name in ['glioma', 'meningioma', 'pituitary']:
        class_count = len(test_df[test_df['class_name'] == class_name])
        mat_count = len(test_df[(test_df['class_name'] == class_name) & (test_df['source'] == 'mat')])
        kaggle_count = len(test_df[(test_df['class_name'] == class_name) & (test_df['source'] == 'kaggle')])
        print(f"   {class_name.capitalize()}: {class_count} (.mat: {mat_count}, kaggle: {kaggle_count})")
    
    print()
    print("💾 Files saved:")
    print(f"   Train CSV: {train_csv_path}")
    print(f"   Test CSV: {test_csv_path}")
    print(f"   Images: {OUTPUT_DIR}/")
    print()
    print("="*80)
    print("✅ DATASET COMBINATION COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Review the combined dataset statistics")
    print("  2. Train a new model using src/models/train_transfer_learning.py")
    print("  3. Update the script to use combined_data_splits instead of data_splits")
    print("="*80)

if __name__ == "__main__":
    combine_datasets()
