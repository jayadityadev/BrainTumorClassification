"""
Brain Tumor Classification - Prediction and Visualization
Includes Grad-CAM visualization for explainability
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from pathlib import Path
import sys
import os
import time

# Import Grad-CAM utilities
sys.path.append(str(Path(__file__).parent))
from gradcam import generate_gradcam

# Simple in-memory cache for loaded Keras models so we don't reload on each request
_MODEL_CACHE = {}


def generate_gradcam(model, img_array, last_conv_layer_name, class_index):
    """
    Generate Grad-CAM heatmap highlighting important regions for prediction.
    
    Args:
        model: Keras model
        img_array: Preprocessed image array (1, H, W, C)
        last_conv_layer_name: Name of last convolutional layer
        class_index: Predicted class index
        
    Returns:
        heatmap: Grad-CAM heatmap (H, W) normalized to [0, 1]
        pred_probs: Prediction probabilities
    """
    # Create a model that outputs last conv layer + predictions
    grad_model = keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Get gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    
    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv outputs by gradients
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()
    
    for i in range(len(pooled_grads)):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # Create heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    if heatmap.max() > 0:
        heatmap /= heatmap.max()  # Normalize to [0, 1]
    
    return heatmap, predictions.numpy()[0]


def predict_with_localization(image_path, model_path, model_type='resnet50', 
                              threshold_percentile=90, save_path=None,
                              confidence_threshold=0.65):
    """
    Predict tumor class and generate visual localization overlay.
    
    Args:
        image_path: Path to input MRI image
        model_path: Path to trained Keras model
        model_type: 'resnet50' or 'densenet121'
        threshold_percentile: Top % of activations to show (default 90 = top 10% only)
        save_path: Where to save visualization (optional)
        confidence_threshold: Minimum confidence to accept prediction (default 0.65 = 65%)
                             Below this, prediction is rejected as uncertain/no tumor
        
    Returns:
        dict with prediction results and visualization
    """
    
    # Label mapping - MUST match training order!
    # Training uses: CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary']
    label_names = {
        0: 'Glioma',
        1: 'Meningioma', 
        2: 'Pituitary Tumor'
    }
    
    # Model architecture config
    if model_type.lower() == 'resnet50':
        base_model_name = 'resnet50'
        conv_layer_name = 'conv5_block3_out'
    elif model_type.lower() == 'densenet121':
        base_model_name = 'densenet121'
        conv_layer_name = 'conv5_block16_concat'
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print("="*80)
    print("🧠 BRAIN TUMOR CLASSIFICATION WITH LOCALIZATION")
    print("="*80)
    print(f"\n📁 Input image: {image_path}")
    print(f"🤖 Model: {model_type}")
    print(f"🎯 Localization threshold: Top {100-threshold_percentile}% activations + >50% intensity\n")
    
    # Load model (cache to avoid reloading on every request)
    print("Loading model...")
    model = None
    try:
        model = _MODEL_CACHE.get(str(model_path))
        if model is not None:
            print(f"Using cached model: {model_path}")
        else:
            t_load = time.time()
            model = keras.models.load_model(model_path)
            _MODEL_CACHE[str(model_path)] = model
            print(f"Loaded model in {time.time() - t_load:.2f}s and cached: {model_path}")
    except Exception as e:
        # Bubble up if load fails
        print(f"Failed to load model at {model_path}: {e}")
        raise

    # Optional GPU sanity check: run a small matmul on /GPU:0 to ensure device is usable
    # Enable by setting environment variable ENABLE_GPU_TEST=1 when starting the server.
    try:
        if os.environ.get('ENABLE_GPU_TEST') == '1':
            print("Running optional GPU sanity test (ENABLE_GPU_TEST=1)")
            try:
                t0 = time.time()
                with tf.device('/GPU:0'):
                    a = tf.random.uniform((512, 512))
                    b = tf.random.uniform((512, 512))
                    c = tf.matmul(a, b)
                    # Force execution and transfer result to host to ensure op ran on GPU
                    _ = c.numpy()
                print(f"✅ GPU sanity test passed (elapsed {time.time()-t0:.3f}s)")
            except Exception as e:
                print("❌ GPU sanity test failed:", e)
    except Exception:
        # Non-fatal: don't break prediction if sanity check errors
        pass
    
    # Access nested base model for Grad-CAM
    try:
        base_model = model.get_layer(base_model_name)
        print(f"✅ Accessed nested {base_model_name} base model")
    except:
        base_model = model
        print(f"⚠️  Using full model (no nested base model found)")
    
    # Load and preprocess image
    print("\n📸 Processing image...")
    # Load as RGB to match training (Kaggle images are RGB)
    img_original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_original is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    original_shape = img_original.shape
    print(f"   Original size: {original_shape}")
    
    # Resize to 128x128 (matching training)
    img_processed = cv2.resize(img_original, (128, 128))
    
    # Normalize to [0, 1] range and add batch dimension
    img_array = np.expand_dims(img_processed.astype(np.float32) / 255.0, axis=0)
    
    print(f"   Processed size: {img_processed.shape}")
    
    # Make prediction
    # Optional: enable TF device placement logging for debugging
    # Set ENABLE_DEVICE_PLACEMENT=1 in the environment to enable verbose device placement logs
    try:
        if os.environ.get('ENABLE_DEVICE_PLACEMENT') == '1':
            tf.debugging.set_log_device_placement(True)
            print("Enabled tf.debugging.set_log_device_placement(True)")
    except Exception:
        pass

    print("\n🔮 Making prediction...")
    # Optional: allow forcing placement on GPU (for debugging) with FORCE_GPU=1
    # Optional timing and GPU-memory reporting for prediction
    # Enable by setting ENABLE_PREDICT_TIMING=1 in the environment
    do_timing = os.environ.get('ENABLE_PREDICT_TIMING') == '1'

    def _get_gpu_mem_info():
        try:
            # returns dict like {'current': <bytes>, 'peak': <bytes>} on TF 2.15+ for GPU:0
            mem = tf.config.experimental.get_memory_info('GPU:0')
            return mem
        except Exception:
            return None

    if os.environ.get('FORCE_GPU') == '1':
        try:
            print("Forcing prediction ops to GPU via tf.device('/GPU:0') (FORCE_GPU=1)")
            if do_timing:
                before_mem = _get_gpu_mem_info()
                t0 = time.time()
                with tf.device('/GPU:0'):
                    pred_probs = model.predict(img_array, verbose=0)[0]
                elapsed = time.time() - t0
                after_mem = _get_gpu_mem_info()
                print(f"model.predict elapsed: {elapsed:.3f}s")
                if before_mem is not None and after_mem is not None:
                    print(f"GPU mem before: {before_mem}, after: {after_mem}")
            else:
                with tf.device('/GPU:0'):
                    pred_probs = model.predict(img_array, verbose=0)[0]
        except Exception as e:
            print("FORCE_GPU prediction failed, falling back to default predict():", e)
            pred_probs = model.predict(img_array, verbose=0)[0]
    else:
        if do_timing:
            before_mem = _get_gpu_mem_info()
            t0 = time.time()
            pred_probs = model.predict(img_array, verbose=0)[0]
            elapsed = time.time() - t0
            after_mem = _get_gpu_mem_info()
            print(f"model.predict elapsed: {elapsed:.3f}s")
            if before_mem is not None and after_mem is not None:
                print(f"GPU mem before: {before_mem}, after: {after_mem}")
        else:
            pred_probs = model.predict(img_array, verbose=0)[0]
    pred_class = np.argmax(pred_probs)
    pred_label = label_names[pred_class]
    confidence = pred_probs[pred_class] * 100
    
    # Check confidence threshold for uncertainty/no tumor detection
    is_uncertain = confidence < (confidence_threshold * 100)
    
    # Calculate prediction entropy (measure of uncertainty)
    # High entropy = model is confused (possible no tumor case)
    epsilon = 1e-10  # Avoid log(0)
    entropy = -np.sum(pred_probs * np.log(pred_probs + epsilon))
    max_entropy = -np.log(1.0 / len(pred_probs))  # Max possible entropy
    normalized_entropy = entropy / max_entropy  # 0 = certain, 1 = maximum confusion
    
    # Additional entropy check: high entropy suggests no clear tumor pattern
    # Entropy > 0.5 means model is quite uncertain about the prediction
    is_high_entropy = normalized_entropy > 0.4
    
    # Mark as uncertain if EITHER low confidence OR high entropy
    is_uncertain = is_uncertain or is_high_entropy
    
    print(f"\n{'='*80}")
    print(f"📊 PREDICTION RESULTS")
    print(f"{'='*80}")
    
    if is_uncertain:
        print(f"⚠️  LOW CONFIDENCE DETECTION")
        print(f"🎯 Top Prediction: {pred_label}")
        print(f"💯 Confidence: {confidence:.2f}% (Below {confidence_threshold*100:.0f}% threshold)")
        print(f"📉 Prediction Entropy: {normalized_entropy:.3f} (0=certain, 1=confused)")
        print(f"\n🚨 RESULT: No definitive tumor detected or image quality issue")
        print(f"   Possible reasons:")
        print(f"   • Normal brain scan (no tumor)")
        print(f"   • Poor image quality")
        print(f"   • Unusual tumor presentation")
        print(f"   • Non-brain image")
        print(f"\n💡 Recommendation: Manual review by radiologist required")
    else:
        print(f"🎯 Predicted Class: {pred_label}")
        print(f"💯 Confidence: {confidence:.2f}%")
        print(f"📉 Prediction Entropy: {normalized_entropy:.3f}")
    
    print(f"\n📈 All Class Probabilities:")
    for class_idx, prob in enumerate(pred_probs):
        print(f"   {label_names[class_idx]}: {prob*100:.2f}%")
    
    # Generate Grad-CAM
    print(f"\n🔥 Generating Grad-CAM localization...")
    try:
        heatmap, _ = generate_gradcam(
            model=base_model,
            img_array=img_array,
            last_conv_layer_name=conv_layer_name,
            class_index=pred_class
        )
        print("✅ Grad-CAM generated successfully")
    except Exception as e:
        print(f"❌ Grad-CAM generation failed: {e}")
        heatmap = np.zeros((128, 128))
    
    # Resize heatmap to match processed image
    if heatmap.shape != (128, 128):
        heatmap = cv2.resize(heatmap, (128, 128))
    
    # Create thresholded mask (top % of activations)
    # Use a more aggressive threshold to show only the most important regions
    threshold = np.percentile(heatmap, threshold_percentile)
    tumor_mask = (heatmap >= threshold).astype(np.uint8)
    
    # Additionally filter out low-intensity activations (< 50% of max)
    max_activation = heatmap.max()
    if max_activation > 0:
        tumor_mask = tumor_mask * (heatmap >= 0.5 * max_activation).astype(np.uint8)
    
    # Post-processing: Refine mask to be more tumor-like
    # 1. Remove small scattered pixels (morphological opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel)
    
    # 2. Fill small holes inside the mask (morphological closing)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. Keep only the largest connected component (main tumor region)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tumor_mask, connectivity=8)
    if num_labels > 1:  # 0 is background
        # Find largest component (excluding background)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        tumor_mask = (labels == largest_component).astype(np.uint8)
    
    # Calculate activated area
    activated_pixels = tumor_mask.sum()
    total_pixels = tumor_mask.size
    activated_percent = (activated_pixels / total_pixels) * 100
    print(f"   Highlighted region: {activated_percent:.1f}% of image")
    
    # Convert to grayscale for CLAHE enhancement
    img_gray = cv2.cvtColor(img_processed, cv2.COLOR_RGB2GRAY)
    
    # Apply contrast enhancement (CLAHE - same as module 2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    fig = plt.figure(figsize=(15, 5))
    
    # Panel 1: Original processed image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img_processed)  # RGB image
    ax1.set_title('Original MRI\n(Preprocessed)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Enhanced image (CLAHE)
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(img_enhanced, cmap='gray')
    ax2.set_title('Enhanced MRI\n(Contrast Enhanced)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Heatmap overlay (semi-transparent)
    ax3 = plt.subplot(1, 3, 3)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_overlay = cv2.addWeighted(img_processed, 0.5, heatmap_colored, 0.5, 0)
    ax3.imshow(heatmap_overlay)
    ax3.set_title('Grad-CAM Overlay\n(Model Attention)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Add overall title with prediction
    if is_uncertain:
        title_color = 'orange'
        title_text = f'⚠️ UNCERTAIN: {pred_label}? ({confidence:.1f}% confidence - Below threshold)'
    else:
        title_color = 'black'
        title_text = f'Prediction: {pred_label} ({confidence:.1f}% confidence)'
    
    fig.suptitle(
        title_text,
        fontsize=16,
        fontweight='bold',
        color=title_color,
        y=1.02
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization: {save_path}")
    else:
        # Auto-save to outputs
        output_dir = Path(__file__).parent.parent.parent / 'outputs' / 'predictions'
        output_dir.mkdir(parents=True, exist_ok=True)
        auto_save_path = output_dir / f"{Path(image_path).stem}_prediction.png"
        plt.savefig(auto_save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization: {auto_save_path}")
    
    print("\n" + "="*80)
    if is_uncertain:
        print("⚠️  UNCERTAIN PREDICTION - REQUIRES REVIEW")
    else:
        print("✅ PREDICTION COMPLETE!")
    print("="*80)
    
    # Return results
    return {
        'predicted_class': pred_label,
        'confidence': confidence,
        'is_uncertain': is_uncertain,
        'entropy': normalized_entropy,
        'confidence_threshold': confidence_threshold * 100,
        'probabilities': {label_names[i]: float(p*100) for i, p in enumerate(pred_probs)},
        'heatmap': heatmap,
        'tumor_mask': tumor_mask,
        'activated_area_percent': activated_percent,
        'visualization_path': save_path if save_path else auto_save_path,
        'warning_message': (
            f"Low confidence ({confidence:.1f}%) - Possible normal brain scan or poor image quality. "
            "Manual review by radiologist required."
        ) if is_uncertain else None
    }


if __name__ == "__main__":
    """
    Example usage - modify these paths for your use case
    """
    
    # Example: Use one of the validation images
    project_root = Path("/projects/ai-ml/BrainTumorProject")
    
    # Load a sample image from validation set
    import pandas as pd
    val_df = pd.read_csv(project_root / "outputs/data_splits/val_split.csv")
    sample_image = val_df.iloc[0]['filepath']
    
    # Model paths
    resnet_model = project_root / "outputs/models/center_crop_fixed/resnet50_center_crop_fixed_20251025_215018.keras"
    densenet_model = project_root / "outputs/models/center_crop_fixed/densenet121_center_crop_fixed_20251025_215730.keras"
    
    print("\n" + "="*80)
    print("🧪 TESTING INFERENCE PIPELINE")
    print("="*80)
    print(f"\nUsing sample image: {Path(sample_image).name}")
    
    # Test with ResNet50
    print("\n\n" + "🔬 Testing with ResNet50" + "\n")
    results = predict_with_localization(
        image_path=sample_image,
        model_path=resnet_model,
        model_type='resnet50',
        threshold_percentile=75
    )
    
    print("\n\n📋 Returned Results:")
    print(f"   Class: {results['predicted_class']}")
    print(f"   Confidence: {results['confidence']:.2f}%")
    print(f"   Highlighted area: {results['activated_area_percent']:.1f}%")
    print(f"   Visualization saved: {results['visualization_path']}")
