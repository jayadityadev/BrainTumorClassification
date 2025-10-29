"""
Comprehensive Training on Combined Dataset (.mat + Kaggle)
Trains both DenseNet121 and ResNet50 for best production results
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121, ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from PIL import Image
from tqdm import tqdm

# Configuration
TRAIN_CSV = 'data/combined_data_splits/train_split.csv'
TEST_CSV = 'data/combined_data_splits/test_split.csv'
OUTPUT_DIR = 'models/current'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LR = 1e-4
NUM_CLASSES = 3
CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary']

# GPU Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

# Force GPU usage with shared access
try:
    # Allow GPU memory to be shared with display server
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        # Set memory growth to avoid OOM and allow shared access
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"✅ GPU ENABLED: Found {len(physical_devices)} GPU(s)")
        for i, gpu in enumerate(physical_devices):
            print(f"   GPU {i}: {gpu.name}")
        
        # Test GPU with a simple operation
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            result = tf.matmul(test_tensor, test_tensor)
            print(f"   GPU Test: ✅ Successfully computed on {result.device}")
    else:
        print("⚠️  No GPU detected - Training will use CPU")
        print("   Note: This will be significantly slower")
except Exception as e:
    print(f"⚠️  GPU configuration error: {e}")
    print("   Attempting to continue with CPU...")

def create_data_generator(csv_path, batch_size, shuffle=True, augment=False):
    """Create data generator from CSV file"""
    df = pd.read_csv(csv_path)
    
    def data_generator():
        indices = np.arange(len(df))
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, len(df), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    row = df.iloc[idx]
                    img_path = row['filepath']
                    label = row['label']
                    
                    try:
                        # Load image (already preprocessed to RGB)
                        img = Image.open(img_path)
                        img_array = np.array(img) / 255.0
                        
                        # Data augmentation for training
                        if augment:
                            # Random horizontal flip
                            if np.random.random() > 0.5:
                                img_array = np.fliplr(img_array)
                            
                            # Random rotation (-10 to +10 degrees)
                            if np.random.random() > 0.5:
                                angle = np.random.uniform(-10, 10)
                                img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                                img_pil = img_pil.rotate(angle, fillcolor=0)
                                img_array = np.array(img_pil) / 255.0
                            
                            # Random brightness adjustment
                            if np.random.random() > 0.5:
                                brightness_factor = np.random.uniform(0.8, 1.2)
                                img_array = np.clip(img_array * brightness_factor, 0, 1)
                        
                        batch_images.append(img_array)
                        batch_labels.append(label)
                        
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
                
                if len(batch_images) > 0:
                    yield (
                        np.array(batch_images, dtype=np.float32),
                        tf.keras.utils.to_categorical(batch_labels, NUM_CLASSES)
                    )
    
    return data_generator

def build_model(base_model_name='densenet121', input_shape=(128, 128, 3)):
    """Build transfer learning model"""
    
    # Load base model
    if base_model_name == 'densenet121':
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model: {base_model_name}")
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model

def train_model(model_name='densenet121'):
    """Train a model on combined dataset"""
    
    print("="*80)
    print(f"🚀 TRAINING {model_name.upper()} ON COMBINED DATASET")
    print("="*80)
    print()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load data info
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"📊 Dataset Statistics:")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Testing samples: {len(test_df)}")
    print()
    
    # Calculate steps
    steps_per_epoch = len(train_df) // BATCH_SIZE
    validation_steps = len(test_df) // BATCH_SIZE
    
    print(f"🔢 Training Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Image size: {IMG_SIZE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    print(f"   Max epochs: {EPOCHS}")
    print()
    
    # Create data generators
    print("📂 Creating data generators...")
    train_generator = create_data_generator(
        TRAIN_CSV, 
        BATCH_SIZE, 
        shuffle=True, 
        augment=True
    )
    val_generator = create_data_generator(
        TEST_CSV, 
        BATCH_SIZE, 
        shuffle=False, 
        augment=False
    )
    
    # Create TF datasets
    train_dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    # Build model
    print("🏗️  Building model...")
    model, base_model = build_model(model_name, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Compile for phase 1 (frozen base)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model built with {model.count_params():,} trainable parameters")
    print()
    
    # ============================================================
    # PHASE 1: Train top layers only (base frozen)
    # ============================================================
    print("="*80)
    print("📚 PHASE 1: Training top layers (base model frozen)")
    print("="*80)
    
    phase1_epochs = 15
    
    callbacks_phase1 = [
        ModelCheckpoint(
            os.path.join(model_output_dir, f'{model_name}_phase1_{timestamp}.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(os.path.join(model_output_dir, f'{model_name}_phase1_training.csv'))
    ]
    
    history_phase1 = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=phase1_epochs,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # ============================================================
    # PHASE 2: Fine-tune entire model (unfreeze base)
    # ============================================================
    print()
    print("="*80)
    print("🔥 PHASE 2: Fine-tuning entire model (base model unfrozen)")
    print("="*80)
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Unfrozen model with {model.count_params():,} trainable parameters")
    print()
    
    phase2_epochs = EPOCHS - phase1_epochs
    
    callbacks_phase2 = [
        ModelCheckpoint(
            os.path.join(model_output_dir, f'{model_name}_phase2_{timestamp}.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1
        ),
        CSVLogger(os.path.join(model_output_dir, f'{model_name}_phase2_training.csv'))
    ]
    
    history_phase2 = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=phase2_epochs,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # ============================================================
    # Save final model
    # ============================================================
    final_model_path = os.path.join(model_output_dir, f'{model_name}_final_{timestamp}.keras')
    model.save(final_model_path)
    print(f"💾 Final model saved: {final_model_path}")
    
    # ============================================================
    # Evaluate on test set
    # ============================================================
    print()
    print("="*80)
    print("📊 EVALUATING MODEL ON TEST SET")
    print("="*80)
    
    # Make predictions
    test_df = pd.read_csv(TEST_CSV)
    y_true = []
    y_pred = []
    
    print("🔮 Making predictions...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        img_path = row['filepath']
        label = row['label']
        
        try:
            img = Image.open(img_path)
            img_array = np.array(img) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            pred = model.predict(img_batch, verbose=0)[0]
            pred_class = np.argmax(pred)
            
            y_true.append(label)
            y_pred.append(pred_class)
            
        except Exception as e:
            print(f"Error predicting {img_path}: {e}")
            continue
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    accuracy = np.mean(y_true == y_pred)
    
    print()
    print("="*80)
    print("📈 RESULTS")
    print("="*80)
    print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%")
    print()
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'{model_name.upper()} - Confusion Matrix\nAccuracy: {accuracy*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(model_output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved: {cm_path}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Combine histories
    all_loss = history_phase1.history['loss'] + history_phase2.history['loss']
    all_val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
    all_acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
    all_val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
    
    plt.subplot(1, 2, 1)
    plt.plot(all_loss, label='Training Loss')
    plt.plot(all_val_loss, label='Validation Loss')
    plt.axvline(x=phase1_epochs, color='red', linestyle='--', label='Fine-tuning starts')
    plt.title(f'{model_name.upper()} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(all_acc, label='Training Accuracy')
    plt.plot(all_val_acc, label='Validation Accuracy')
    plt.axvline(x=phase1_epochs, color='red', linestyle='--', label='Fine-tuning starts')
    plt.title(f'{model_name.upper()} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    history_path = os.path.join(model_output_dir, f'{model_name}_training_history.png')
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    print(f"💾 Training history saved: {history_path}")
    
    print()
    print("="*80)
    print(f"✅ {model_name.upper()} TRAINING COMPLETE!")
    print("="*80)
    print()
    
    return model, accuracy, final_model_path

def main():
    """Main training function"""
    
    print("="*80)
    print("🧠 COMPREHENSIVE TRAINING ON COMBINED DATASET")
    print("="*80)
    print()
    print("📊 Dataset: .mat files + Kaggle images")
    print("🎯 Goal: Best production model with maximum generalization")
    print()
    print("🔬 Training Strategy:")
    print("   1. Train DenseNet121")
    print("   2. Train ResNet50")
    print("   3. Compare and select best model")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {}
    
    # Train DenseNet121
    print("\n" + "="*80)
    print("🏃 STARTING DENSENET121 TRAINING")
    print("="*80)
    model_densenet, acc_densenet, path_densenet = train_model('densenet121')
    results['densenet121'] = {'accuracy': acc_densenet, 'path': path_densenet}
    
    # Train ResNet50
    print("\n" + "="*80)
    print("🏃 STARTING RESNET50 TRAINING")
    print("="*80)
    model_resnet, acc_resnet, path_resnet = train_model('resnet50')
    results['resnet50'] = {'accuracy': acc_resnet, 'path': path_resnet}
    
    # Summary
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*80)
    print()
    print("📊 Model Comparison:")
    print(f"   DenseNet121: {acc_densenet*100:.2f}%")
    print(f"   ResNet50: {acc_resnet*100:.2f}%")
    print()
    
    best_model = 'densenet121' if acc_densenet > acc_resnet else 'resnet50'
    best_acc = max(acc_densenet, acc_resnet)
    
    print(f"🥇 Best Model: {best_model.upper()} ({best_acc*100:.2f}%)")
    print(f"📁 Best Model Path: {results[best_model]['path']}")
    print()
    
    # Save summary
    summary_df = pd.DataFrame([
        {
            'model': 'densenet121',
            'accuracy': acc_densenet,
            'path': path_densenet
        },
        {
            'model': 'resnet50',
            'accuracy': acc_resnet,
            'path': path_resnet
        }
    ])
    
    summary_path = os.path.join(OUTPUT_DIR, 'training_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Summary saved: {summary_path}")
    
    print()
    print("="*80)
    print("✅ ALL TRAINING COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Update your inference scripts to use the best model")
    print("  2. Test on Kaggle test set to verify generalization")
    print("  3. Deploy the best model to your web application")
    print("="*80)

if __name__ == "__main__":
    main()
