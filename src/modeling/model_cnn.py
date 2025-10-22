"""
CNN Model Architecture for Brain Tumor Classification

This module defines the CNN architecture used for classifying
brain tumors into three categories: Meningioma, Glioma, and Pituitary.

Author: Jayaditya Dev
Date: October 22, 2025
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam


def build_cnn_model(
    input_shape: tuple = (128, 128, 1),
    num_classes: int = 3,
    learning_rate: float = 1e-4
):
    """
    Build CNN model for brain tumor classification.
    
    Architecture:
        Input (128, 128, 1)
        ↓
        [Conv Block 1] → 32 filters, 3×3
        ↓ MaxPool (64, 64, 32)
        [Conv Block 2] → 64 filters, 3×3
        ↓ MaxPool (32, 32, 64)
        [Conv Block 3] → 128 filters, 3×3
        ↓ MaxPool (16, 16, 128)
        [Flatten] → 32,768 features
        ↓
        [Dense] → 128 neurons + Dropout
        ↓
        [Output] → 3 classes (Softmax)
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, 
               padding='same', name='conv1'),
        MaxPooling2D((2, 2), name='pool1'),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        MaxPooling2D((2, 2), name='pool2'),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        MaxPooling2D((2, 2), name='pool3'),
        
        # Flatten and Dense Layers
        Flatten(name='flatten'),
        Dense(128, activation='relu', name='dense1'),
        Dropout(0.5, name='dropout'),
        Dense(num_classes, activation='softmax', name='output')
    ], name='BrainTumorCNN')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model built and compiled successfully")
    print(f"   Input shape: {input_shape}")
    print(f"   Output classes: {num_classes}")
    print(f"   Learning rate: {learning_rate}")
    
    return model


def print_model_info(model):
    """
    Print detailed model information.
    
    Args:
        model: Keras model
    """
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    
    model.summary()
    
    total_params = model.count_params()
    print(f"\n{'='*60}")
    print("PARAMETER DETAILS")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    # Layer-by-layer breakdown
    print(f"\n{'='*60}")
    print("LAYER-BY-LAYER BREAKDOWN")
    print("="*60)
    
    for i, layer in enumerate(model.layers, 1):
        layer_params = layer.count_params()
        print(f"{i}. {layer.name:15s} | Output: {str(layer.output.shape):25s} | Params: {layer_params:,}")
    
    print("="*60)


def enable_gpu_memory_growth():
    """
    Enable GPU memory growth to prevent TensorFlow from allocating
    all GPU memory at once. Important for GTX 1650 (4GB VRAM).
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"❌ Error enabling memory growth: {e}")
    else:
        print("⚠️ No GPU detected. Running on CPU.")


if __name__ == "__main__":
    # Example usage
    print("CNN Model Module - Example Usage\n")
    
    # Enable GPU memory growth
    enable_gpu_memory_growth()
    
    # Check TensorFlow and GPU
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    
    # Build model
    model = build_cnn_model(
        input_shape=(128, 128, 1),
        num_classes=3,
        learning_rate=1e-4
    )
    
    # Print detailed info
    print_model_info(model)
    
    print("\n✅ Model created successfully!")
