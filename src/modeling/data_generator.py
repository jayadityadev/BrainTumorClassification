"""
Data Generator Utilities for Brain Tumor Classification

This module provides functions to create Keras ImageDataGenerators
for training, validation, and testing with proper augmentation.

Author: Jayaditya Dev
Date: October 22, 2025
"""

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_train_generator(
    csv_path: str,
    batch_size: int = 32,
    target_size: tuple = (128, 128),
    seed: int = 42
):
    """
    Create training data generator with augmentation.
    
    Args:
        csv_path: Path to train CSV file with 'filepath' and 'label' columns
        batch_size: Number of images per batch
        target_size: Target image size (height, width)
        seed: Random seed for reproducibility
        
    Returns:
        Keras DirectoryIterator for training data
    """
    # Load training metadata
    train_df = pd.read_csv(csv_path)
    train_df['label'] = train_df['label'].astype(str)
    
    # Create augmented data generator
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalize to [0, 1]
        rotation_range=15,           # Random rotation ±15 degrees
        width_shift_range=0.05,      # Horizontal shift up to 5%
        height_shift_range=0.05,     # Vertical shift up to 5%
        zoom_range=0.1,              # Zoom in/out up to 10%
        horizontal_flip=True,        # Random horizontal flip
        vertical_flip=False,         # NO vertical flip (anatomically invalid)
        fill_mode='nearest'          # Fill mode for transformations
    )
    
    # Create generator
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )
    
    print(f"✅ Training generator created")
    print(f"   Images: {len(train_df)}")
    print(f"   Batches: {len(train_generator)}")
    print(f"   Classes: {train_generator.class_indices}")
    
    return train_generator


def create_val_test_generator(
    csv_path: str,
    batch_size: int = 32,
    target_size: tuple = (128, 128),
    shuffle: bool = False,
    seed: int = 42
):
    """
    Create validation/test data generator (no augmentation).
    
    Args:
        csv_path: Path to val/test CSV file with 'filepath' and 'label' columns
        batch_size: Number of images per batch
        target_size: Target image size (height, width)
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        
    Returns:
        Keras DirectoryIterator for validation/test data
    """
    # Load metadata
    df = pd.read_csv(csv_path)
    df['label'] = df['label'].astype(str)
    
    # Create data generator (only rescaling, no augmentation)
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generator
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filepath',
        y_col='label',
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed
    )
    
    print(f"✅ Generator created")
    print(f"   Images: {len(df)}")
    print(f"   Batches: {len(generator)}")
    print(f"   Shuffle: {shuffle}")
    
    return generator


if __name__ == "__main__":
    # Example usage
    print("Data Generator Module - Example Usage\n")
    
    # Create training generator
    train_gen = create_train_generator(
        csv_path='../../outputs/data_splits/train_split.csv',
        batch_size=32
    )
    
    # Create validation generator
    val_gen = create_val_test_generator(
        csv_path='../../outputs/data_splits/val_split.csv',
        batch_size=32,
        shuffle=False
    )
    
    # Create test generator
    test_gen = create_val_test_generator(
        csv_path='../../outputs/data_splits/test_split.csv',
        batch_size=32,
        shuffle=False
    )
    
    print("\n✅ All generators created successfully!")
