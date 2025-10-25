"""
Helper utilities for advanced model training with transfer learning.

This module provides utilities for:
- Converting grayscale images to RGB
- Creating custom data generators
- Model ensemble predictions
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence


class GrayscaleToRGBGenerator(Sequence):
    """
    Custom data generator that converts grayscale images to RGB.
    
    Pretrained models (ResNet, EfficientNet, etc.) require RGB input (3 channels),
    but our brain MRI images are grayscale (1 channel). This generator converts
    grayscale to RGB by replicating the grayscale channel 3 times.
    
    Inherits from keras.utils.Sequence for proper integration with model.fit()
    """
    
    def __init__(self, base_generator):
        """
        Args:
            base_generator: Keras ImageDataGenerator flow_from_dataframe generator
        """
        self.base_generator = base_generator
        self.batch_size = base_generator.batch_size
        self.n = base_generator.n
        self.batch_index = 0
        
    def __len__(self):
        """Number of batches in the generator."""
        return len(self.base_generator)
    
    def __getitem__(self, index):
        """Get batch at index with RGB conversion."""
        # Get batch from base generator using index
        batch_x, batch_y = self.base_generator[index]
        
        # Convert grayscale (H, W, 1) to RGB (H, W, 3)
        # Replicate the single channel 3 times
        if batch_x.shape[-1] == 1:
            batch_x = np.repeat(batch_x, 3, axis=-1)
        
        return batch_x, batch_y
    
    def reset(self):
        """Reset the generator."""
        self.base_generator.reset()
    
    @property
    def classes(self):
        """Get classes from base generator."""
        return self.base_generator.classes
    
    @property
    def class_indices(self):
        """Get class indices from base generator."""
        return self.base_generator.class_indices


def create_rgb_generators(train_df, val_df, test_df, img_size=(128, 128), batch_size=32):
    """
    Create data generators that output RGB images for transfer learning.
    
    Args:
        train_df: Training dataframe with 'filepath' and 'label' columns
        val_df: Validation dataframe
        test_df: Test dataframe
        img_size: Target image size (height, width)
        batch_size: Batch size
    
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Training augmentation
    # NOTE: No rescale - EfficientNet expects images in [0, 255] range
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,  # Safe for MRI
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect'
    )
    
    # Validation/Test (no augmentation)
    # NOTE: No rescale - EfficientNet expects images in [0, 255] range
    val_test_datagen = ImageDataGenerator()
    
    # Create base generators (grayscale)
    train_gen_base = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',  # Load as grayscale
        shuffle=True,
        seed=42
    )
    
    val_gen_base = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    
    test_gen_base = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    
    # Wrap with RGB converter
    train_generator = GrayscaleToRGBGenerator(train_gen_base)
    val_generator = GrayscaleToRGBGenerator(val_gen_base)
    test_generator = GrayscaleToRGBGenerator(test_gen_base)
    
    return train_generator, val_generator, test_generator


def ensemble_predict(models, generator, voting='soft'):
    """
    Make ensemble predictions using multiple models.
    
    Args:
        models: List of trained Keras models
        generator: Data generator
        voting: 'soft' (average probabilities) or 'hard' (majority vote)
    
    Returns:
        numpy.ndarray: Ensemble predictions
    """
    # Get predictions from all models
    all_predictions = []
    
    for i, model in enumerate(models):
        print(f"Getting predictions from model {i+1}/{len(models)}...")
        generator.reset()
        predictions = model.predict(generator, verbose=0)
        all_predictions.append(predictions)
    
    # Combine predictions
    if voting == 'soft':
        # Average probabilities
        ensemble_pred = np.mean(all_predictions, axis=0)
    else:
        # Majority vote
        # Get class predictions from each model
        class_predictions = [np.argmax(pred, axis=1) for pred in all_predictions]
        # Stack and take mode (most common prediction)
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=0, 
            arr=np.array(class_predictions)
        )
    
    return ensemble_pred


def test_time_augmentation(model, generator, n_augmentations=5):
    """
    Test-time augmentation: predict on augmented versions and average.
    
    Args:
        model: Trained Keras model
        generator: Data generator with augmentation enabled
        n_augmentations: Number of augmented predictions to average
    
    Returns:
        numpy.ndarray: Averaged predictions
    """
    all_predictions = []
    
    for i in range(n_augmentations):
        generator.reset()
        predictions = model.predict(generator, verbose=0)
        all_predictions.append(predictions)
    
    # Average all predictions
    return np.mean(all_predictions, axis=0)


if __name__ == '__main__':
    print("Transfer learning utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - GrayscaleToRGBGenerator: Convert grayscale to RGB")
    print("  - create_rgb_generators: Create RGB data generators")
    print("  - ensemble_predict: Ensemble predictions from multiple models")
    print("  - test_time_augmentation: TTA for better accuracy")
