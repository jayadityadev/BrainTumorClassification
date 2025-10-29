"""
Evaluate model performance on Kaggle test set (kaggle_temp/Testing)
This measures cross-domain generalization from .mat-trained model to Kaggle JPGs
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'models/current/densenet121_finetuned.keras'
KAGGLE_TEST_DIR = PROJECT_ROOT / 'kaggle_temp/Testing'
OUTPUT_DIR = PROJECT_ROOT / 'outputs/reports'
IMG_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.80

# Class mapping
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary']
CLASS_TO_LABEL = {'glioma': 0, 'meningioma': 1, 'pituitary': 2}

def preprocess_image(image_path):
    """Preprocess image using center-crop method"""
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Center crop
    h, w = img_array.shape
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    img_cropped = img_array[start_h:start_h+min_dim, start_w:start_w+min_dim]
    
    # Resize and normalize
    img_resized = Image.fromarray(img_cropped).resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_normalized = np.array(img_resized) / 255.0
    
    # Convert to RGB (3 channels) for DenseNet
    img_rgb = np.stack([img_normalized] * 3, axis=-1)
    
    return img_rgb

def calculate_entropy(probs):
    """Calculate normalized entropy"""
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    return entropy / max_entropy if max_entropy > 0 else 0

def load_model_with_nested_access(model_path):
    """Load model and access nested base model if needed"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Check if nested
    if hasattr(model, 'layers') and len(model.layers) > 0:
        if hasattr(model.layers[0], 'layers'):
            print("✅ Accessing nested base model")
            model = model.layers[0]
    
    return model

def evaluate_kaggle_test():
    """Evaluate model on Kaggle test set"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("🧪 KAGGLE TEST SET EVALUATION")
    print("="*80)
    print()
    
    # Load model
    model = load_model_with_nested_access(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
    print()
    
    # Collect all test images
    all_images = []
    all_labels = []
    all_filepaths = []
    
    print("📂 Loading test images...")
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(KAGGLE_TEST_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️  Warning: {class_dir} not found, skipping...")
            continue
        
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"   {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            all_filepaths.append(img_path)
            all_labels.append(CLASS_TO_LABEL[class_name])
    
    # Skip 'notumor' class as we didn't train on it
    print(f"   ⚠️  Skipping 'notumor' class (not in training data)")
    
    print(f"\n📊 Total test images (excluding notumor): {len(all_filepaths)}")
    print()
    
    # Make predictions
    print("🔮 Making predictions...")
    predictions = []
    confidences = []
    entropies = []
    uncertain_flags = []
    
    for img_path in tqdm(all_filepaths, desc="Processing"):
        try:
            # Preprocess
            img_array = preprocess_image(img_path)
            img_batch = np.expand_dims(img_array, axis=0)  # Already has 3 channels
            
            # Predict
            probs = model.predict(img_batch, verbose=0)[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            entropy = calculate_entropy(probs)
            
            predictions.append(pred_class)
            confidences.append(confidence)
            entropies.append(entropy)
            
            # Check uncertainty
            is_uncertain = (confidence < CONFIDENCE_THRESHOLD) or (entropy > 0.4)
            uncertain_flags.append(is_uncertain)
            
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
            predictions.append(-1)
            confidences.append(0.0)
            entropies.append(1.0)
            uncertain_flags.append(True)
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(predictions)
    confidences = np.array(confidences)
    entropies = np.array(entropies)
    uncertain_flags = np.array(uncertain_flags)
    
    # Filter out errors
    valid_mask = y_pred >= 0
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    confidences = confidences[valid_mask]
    entropies = entropies[valid_mask]
    uncertain_flags = uncertain_flags[valid_mask]
    all_filepaths = [fp for i, fp in enumerate(all_filepaths) if valid_mask[i]]
    
    print()
    print("="*80)
    print("📊 OVERALL RESULTS")
    print("="*80)
    
    # Overall accuracy
    overall_accuracy = np.mean(y_true == y_pred)
    print(f"\n🎯 Overall Accuracy: {overall_accuracy*100:.2f}%")
    
    # Confident vs Uncertain predictions
    confident_mask = ~uncertain_flags
    if np.sum(confident_mask) > 0:
        confident_accuracy = np.mean(y_true[confident_mask] == y_pred[confident_mask])
        print(f"✅ Confident Predictions ({np.sum(confident_mask)}/{len(y_pred)}): {confident_accuracy*100:.2f}%")
    
    if np.sum(uncertain_flags) > 0:
        uncertain_accuracy = np.mean(y_true[uncertain_flags] == y_pred[uncertain_flags])
        print(f"⚠️  Uncertain Predictions ({np.sum(uncertain_flags)}/{len(y_pred)}): {uncertain_accuracy*100:.2f}%")
    
    print(f"\n📊 Confidence Stats:")
    print(f"   Mean: {np.mean(confidences)*100:.2f}%")
    print(f"   Median: {np.median(confidences)*100:.2f}%")
    print(f"   Std: {np.std(confidences)*100:.2f}%")
    
    print(f"\n📊 Entropy Stats:")
    print(f"   Mean: {np.mean(entropies):.3f}")
    print(f"   Median: {np.median(entropies):.3f}")
    print(f"   Std: {np.std(entropies):.3f}")
    
    # Classification report
    print("\n" + "="*80)
    print("📈 CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - Kaggle Test Set\nOverall Accuracy: {overall_accuracy*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_kaggle_test.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Confusion matrix saved: {cm_path}")
    
    # Plot confidence distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(confidences[y_true == y_pred], bins=30, alpha=0.7, label='Correct', color='green')
    plt.hist(confidences[y_true != y_pred], bins=30, alpha=0.7, label='Incorrect', color='red')
    plt.axvline(x=CONFIDENCE_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({CONFIDENCE_THRESHOLD})')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(entropies[y_true == y_pred], bins=30, alpha=0.7, label='Correct', color='green')
    plt.hist(entropies[y_true != y_pred], bins=30, alpha=0.7, label='Incorrect', color='red')
    plt.axvline(x=0.4, color='black', linestyle='--', label='Threshold (0.4)')
    plt.xlabel('Entropy')
    plt.ylabel('Count')
    plt.title('Entropy Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    dist_path = os.path.join(OUTPUT_DIR, 'confidence_entropy_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"💾 Distribution plots saved: {dist_path}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'filepath': all_filepaths,
        'true_label': [CLASS_NAMES[i] for i in y_true],
        'predicted_label': [CLASS_NAMES[i] for i in y_pred],
        'confidence': confidences,
        'entropy': entropies,
        'is_uncertain': uncertain_flags,
        'is_correct': y_true == y_pred
    })
    
    results_path = os.path.join(OUTPUT_DIR, 'kaggle_test_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"💾 Detailed results saved: {results_path}")
    
    # Find worst predictions
    print("\n" + "="*80)
    print("❌ TOP 10 WORST PREDICTIONS (Incorrect + High Confidence)")
    print("="*80)
    incorrect_df = results_df[results_df['is_correct'] == False].copy()
    if len(incorrect_df) > 0:
        incorrect_df = incorrect_df.sort_values('confidence', ascending=False).head(10)
        for idx, row in incorrect_df.iterrows():
            filename = os.path.basename(row['filepath'])
            print(f"   {filename}")
            print(f"      True: {row['true_label']} | Pred: {row['predicted_label']} | Conf: {row['confidence']*100:.2f}% | Entropy: {row['entropy']:.3f}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nKey Finding: Cross-domain accuracy is {overall_accuracy*100:.2f}%")
    print(f"             (vs 95.91% on in-distribution .mat test set)")
    print(f"             Domain shift impact: {(95.91 - overall_accuracy*100):.2f}% accuracy drop")
    print("\n💡 Recommendation: Combine both datasets for retraining to improve generalization")
    print("="*80)

if __name__ == "__main__":
    evaluate_kaggle_test()
