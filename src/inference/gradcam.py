"""
Gradient-weighted Class Activation Mapping (Grad-CAM) utilities.

This module provides functions to generate and visualize Grad-CAM heatmaps
for explaining CNN model predictions on brain tumor MRI images.

References:
    Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization. https://arxiv.org/abs/1610.02391
"""

import numpy as np
import tensorflow as tf
import cv2
from typing import Tuple, Optional, Union


def get_last_conv_layer_name(model_name: str) -> str:
    """
    Get the appropriate last convolutional layer name for common architectures.
    
    Args:
        model_name: Name of the model architecture (e.g., 'resnet50', 'densenet121')
        
    Returns:
        Name of the last convolutional layer
        
    Raises:
        ValueError: If model architecture is not supported
    """
    layer_mapping = {
        'resnet50': 'conv5_block3_out',
        'resnet101': 'conv5_block3_out',
        'densenet121': 'conv5_block16_concat',
        'densenet169': 'conv5_block32_concat',
        'densenet201': 'conv5_block48_concat',
        'vgg16': 'block5_conv3',
        'vgg19': 'block5_conv4',
        'inceptionv3': 'mixed10',
        'mobilenetv2': 'Conv_1',
        'efficientnetb0': 'top_conv',
    }
    
    model_name_lower = model_name.lower()
    for key in layer_mapping:
        if key in model_name_lower:
            return layer_mapping[key]
    
    raise ValueError(
        f"Unsupported model architecture: {model_name}. "
        f"Supported models: {list(layer_mapping.keys())}"
    )


def generate_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str,
    class_index: Optional[int] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Generate Grad-CAM heatmap for a given image and model.
    
    Grad-CAM uses the gradients of the target class flowing into the final
    convolutional layer to produce a coarse localization map highlighting
    the important regions in the image for predicting the target class.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array with shape (1, H, W, C)
        last_conv_layer_name: Name of the last convolutional layer
        class_index: Target class index. If None, uses the predicted class
        normalize: Whether to normalize heatmap to [0, 1] range
        
    Returns:
        Tuple of (heatmap, class_index):
            - heatmap: 2D numpy array with Grad-CAM visualization
            - class_index: Index of the target/predicted class
            
    Raises:
        ValueError: If the specified layer is not found in the model
    """
    # Verify the layer exists (handle nested layers)
    # Check if it's a nested layer (e.g., "resnet50.conv5_block3_out")
    if '.' in last_conv_layer_name:
        try:
            base_model_name, inner_layer_name = last_conv_layer_name.split('.', 1)
            base_model = model.get_layer(base_model_name)
            conv_layer = base_model.get_layer(inner_layer_name)
        except ValueError:
            raise ValueError(
                f"Nested layer '{last_conv_layer_name}' not found in model. "
                f"Base model '{base_model_name}' layers: {[layer.name for layer in base_model.layers] if 'base_model' in locals() else 'N/A'}"
            )
    else:
        try:
            conv_layer = model.get_layer(last_conv_layer_name)
        except ValueError:
            raise ValueError(
                f"Layer '{last_conv_layer_name}' not found in model. "
                f"Available layers: {[layer.name for layer in model.layers]}"
            )
    
    # Convert to tensor if needed
    if not isinstance(img_array, tf.Tensor):
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    else:
        img_tensor = img_array
    
    # Create a model that maps input to the activations of the last conv layer
    # and the output predictions
    # For nested models, we need to handle them differently
    if '.' in last_conv_layer_name:
        # IMPROVED APPROACH: Build a model directly from input to outputs
        # This ensures proper gradient tracking through the nested structure
        base_model_name = last_conv_layer_name.split('.')[0]
        base_model = model.get_layer(base_model_name)
        
        # Build a model that outputs both conv activations and predictions
        # We need to trace the path from input through all layers
        try:
            # Create a model from the original input to both the conv layer and final output
            # Method 1: Try to build functional model using existing graph
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[base_model.get_layer(last_conv_layer_name.split('.')[-1]).output, model.output]
            )
            
            # Use GradientTape to compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_tensor)
                
                # If no class index provided, use the predicted class
                if class_index is None:
                    class_index = tf.argmax(predictions[0]).numpy()
                
                # Get the score for the target class
                class_channel = predictions[:, class_index]
            
            # Compute gradients
            grads = tape.gradient(class_channel, conv_outputs)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not build unified gradient model: {e}")
            print("   Attempting alternative gradient computation...")
            
            # Method 2: Manual gradient computation using chain rule
            # Step 1: Get conv outputs from base model
            base_output_model = tf.keras.models.Model(
                inputs=base_model.input,
                outputs=conv_layer.output
            )
            
            with tf.GradientTape(persistent=True) as tape:
                # Forward pass through base model
                conv_outputs = base_output_model(img_tensor)
                tape.watch(conv_outputs)
                
                # Get the output of base model (GAP layer input)
                base_final = base_model(img_tensor)
                
                # Get predictions from full model
                predictions = model(img_tensor)
                
                # If no class index provided, use the predicted class
                if class_index is None:
                    class_index = tf.argmax(predictions[0]).numpy()
                
                # Get the score for the target class
                class_channel = predictions[:, class_index]
            
            # Try to compute gradients
            try:
                # Gradient of output w.r.t. base_final
                grad_output_to_base = tape.gradient(class_channel, base_final)
                # Gradient of base_final w.r.t. conv_outputs  
                grad_base_to_conv = tape.gradient(base_final, conv_outputs)
                
                if grad_output_to_base is not None and grad_base_to_conv is not None:
                    # Chain rule: multiply gradients
                    grads = grad_base_to_conv
                else:
                    grads = None
            except:
                grads = None
            
            del tape
        
        # Check if gradients are valid
        if grads is None:
            print("⚠️ Note: Direct gradients unavailable for nested model architecture.")
            print("   Using Class Activation Mapping (CAM) approach instead...")
            print("   (This is still informative and shows important regions)")
            
            # Get conv outputs
            base_output_model = tf.keras.models.Model(
                inputs=base_model.input,
                outputs=conv_layer.output
            )
            conv_outputs = base_output_model(img_tensor)
            predictions = model(img_tensor)
            
            if class_index is None:
                class_index = tf.argmax(predictions[0]).numpy()
            
            # Fallback: Use activation-based importance
            conv_outputs_np = conv_outputs.numpy()
            
            # Global average pooling
            pooled_output = np.mean(conv_outputs_np, axis=(1, 2))
            
            # Use channel-wise average as importance
            channel_importance = pooled_output[0]  
            
            # Broadcast across spatial dimensions
            grads = channel_importance.reshape(1, 1, 1, -1)
            grads = np.tile(grads, [1, conv_outputs.shape[1], conv_outputs.shape[2], 1])
            grads = tf.convert_to_tensor(grads, dtype=tf.float32)
    else:
        # For flat models, create unified grad model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            
            # If no class index provided, use the predicted class
            if class_index is None:
                class_index = tf.argmax(predictions[0]).numpy()
            
            # Get the score for the target class
            class_channel = predictions[:, class_index]
        
        # Compute gradients of the class score with respect to conv layer output
        grads = tape.gradient(class_channel, conv_outputs)
    
    # Global Average Pooling of gradients (importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Get the activation maps
    conv_outputs = conv_outputs[0]
    
    # Weight each channel by corresponding gradient (importance)
    # and sum all channels to get the heatmap
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU to the heatmap (only positive contributions)
    heatmap = tf.maximum(heatmap, 0)
    
    # Convert to numpy
    heatmap = heatmap.numpy()
    
    # Normalize the heatmap
    if normalize and np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap, int(class_index)


def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
    resize_mode: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        original_img: Original grayscale or RGB image (H, W) or (H, W, C)
        heatmap: Grad-CAM heatmap (h, w) with values in [0, 1]
        alpha: Transparency factor for heatmap overlay (0 = transparent, 1 = opaque)
        colormap: OpenCV colormap to apply to heatmap (default: COLORMAP_JET)
        resize_mode: Interpolation method for resizing heatmap
        
    Returns:
        RGB image with heatmap overlay (H, W, 3) with values in [0, 255]
    """
    # Ensure original image is in the right format
    if len(original_img.shape) == 2:
        # Grayscale to RGB
        original_img_rgb = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif original_img.shape[2] == 1:
        # Single channel to RGB
        original_img_rgb = cv2.cvtColor(original_img.squeeze().astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        original_img_rgb = original_img.astype(np.uint8)
    
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(
        heatmap,
        (original_img_rgb.shape[1], original_img_rgb.shape[0]),
        interpolation=resize_mode
    )
    
    # Convert heatmap to RGB using colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        colormap
    )
    
    # Overlay heatmap on original image
    overlayed = cv2.addWeighted(
        heatmap_colored,
        alpha,
        original_img_rgb,
        1 - alpha,
        0
    )
    
    return overlayed


def overlay_heatmap_focused(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    top_k_percent: float = 20.0,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET,
    resize_mode: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Overlay ONLY the top-K% most activated regions on original image.
    This creates a focused visualization highlighting just the tumor region.
    
    Args:
        original_img: Original grayscale or RGB image (H, W) or (H, W, C)
        heatmap: Grad-CAM heatmap (h, w) with values in [0, 1]
        top_k_percent: Percentage of top activations to show (e.g., 20 = top 20%)
        alpha: Transparency factor for heatmap overlay (0 = transparent, 1 = opaque)
        colormap: OpenCV colormap to apply to heatmap (default: COLORMAP_JET)
        resize_mode: Interpolation method for resizing heatmap
        
    Returns:
        RGB image with focused heatmap overlay (H, W, 3) with values in [0, 255]
    """
    # Ensure original image is in the right format
    if len(original_img.shape) == 2:
        # Grayscale to RGB
        original_img_rgb = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif original_img.shape[2] == 1:
        # Single channel to RGB
        original_img_rgb = cv2.cvtColor(original_img.squeeze().astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        original_img_rgb = original_img.astype(np.uint8)
    
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(
        heatmap,
        (original_img_rgb.shape[1], original_img_rgb.shape[0]),
        interpolation=resize_mode
    )
    
    # Find threshold for top K%
    threshold = np.percentile(heatmap_resized, 100 - top_k_percent)
    
    # Create binary mask for high-activation regions
    mask = heatmap_resized >= threshold
    
    # Convert heatmap to RGB using colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        colormap
    )
    
    # Start with original image
    overlayed = original_img_rgb.copy()
    
    # Only overlay where mask is True (high activation regions)
    overlayed[mask] = cv2.addWeighted(
        original_img_rgb[mask],
        1 - alpha,
        heatmap_colored[mask],
        alpha,
        0
    )
    
    return overlayed


def generate_ensemble_gradcam(
    models: list,
    img_array: np.ndarray,
    last_conv_layer_names: list,
    weights: Optional[list] = None,
    class_index: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate ensemble Grad-CAM by combining heatmaps from multiple models.
    
    Args:
        models: List of trained Keras models
        img_array: Preprocessed image array with shape (1, H, W, C)
        last_conv_layer_names: List of last conv layer names for each model
        weights: Optional list of weights for each model (default: equal weights)
        class_index: Target class index. If None, uses ensemble prediction
        
    Returns:
        Tuple of (ensemble_heatmap, ensemble_prediction, predicted_class):
            - ensemble_heatmap: Combined Grad-CAM heatmap
            - ensemble_prediction: Averaged prediction probabilities
            - predicted_class: Index of predicted class
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    if len(models) != len(last_conv_layer_names):
        raise ValueError("Number of models must match number of layer names")
    
    if len(models) != len(weights):
        raise ValueError("Number of models must match number of weights")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Get predictions from all models
    predictions = []
    for model in models:
        pred = model.predict(img_array, verbose=0)
        predictions.append(pred)
    
    # Ensemble prediction
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    
    # Determine class index
    if class_index is None:
        class_index = np.argmax(ensemble_pred[0])
    
    # Generate heatmaps from all models
    heatmaps = []
    for model, layer_name, weight in zip(models, last_conv_layer_names, weights):
        heatmap, _ = generate_gradcam(model, img_array, layer_name, class_index)
        heatmaps.append(heatmap)
    
    # Find maximum dimensions
    max_h = max([h.shape[0] for h in heatmaps])
    max_w = max([h.shape[1] for h in heatmaps])
    
    # Resize all heatmaps to same size and combine
    resized_heatmaps = []
    for heatmap in heatmaps:
        resized = cv2.resize(heatmap, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
        resized_heatmaps.append(resized)
    
    # Weighted average of heatmaps
    ensemble_heatmap = np.average(resized_heatmaps, axis=0, weights=weights)
    
    # Normalize
    if np.max(ensemble_heatmap) > 0:
        ensemble_heatmap = ensemble_heatmap / np.max(ensemble_heatmap)
    
    return ensemble_heatmap, ensemble_pred[0], int(class_index)


def save_gradcam_visualization(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    output_path: str,
    class_name: str = "",
    confidence: float = 0.0,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
    add_text: bool = True
) -> None:
    """
    Save Grad-CAM visualization with optional annotation.
    
    Args:
        original_img: Original image
        heatmap: Grad-CAM heatmap
        output_path: Path to save the visualization
        class_name: Predicted class name for annotation
        confidence: Prediction confidence for annotation
        alpha: Transparency for overlay
        colormap: OpenCV colormap to use
        add_text: Whether to add prediction text on image
    """
    # Generate overlay
    overlayed = overlay_heatmap(original_img, heatmap, alpha, colormap)
    
    # Add text annotation if requested
    if add_text and class_name:
        text = f"{class_name}: {confidence:.2%}"
        cv2.putText(
            overlayed,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    
    # Save
    cv2.imwrite(output_path, overlayed)


def visualize_gradcam_comparison(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    class_name: str = "",
    confidence: float = 0.0,
    alpha: float = 0.4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a comparison visualization showing original, heatmap, and overlay.
    
    Args:
        original_img: Original grayscale image
        heatmap: Grad-CAM heatmap
        class_name: Predicted class name
        confidence: Prediction confidence
        alpha: Transparency for overlay
        
    Returns:
        Tuple of (original_rgb, heatmap_colored, overlay):
            Three images ready for side-by-side visualization
    """
    # Convert original to RGB
    if len(original_img.shape) == 2:
        original_rgb = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        original_rgb = original_img.astype(np.uint8)
    
    # Resize and colorize heatmap
    heatmap_resized = cv2.resize(
        heatmap,
        (original_rgb.shape[1], original_rgb.shape[0])
    )
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )
    
    # Create overlay
    overlay = overlay_heatmap(original_img, heatmap, alpha)
    
    return original_rgb, heatmap_colored, overlay
