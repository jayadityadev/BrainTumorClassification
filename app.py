"""
Brain Tumor Classifier - Web Interface
Simple Flask app for uploading MRI scans and getting predictions
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import sys
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src' / 'inference'))
from predict import predict_with_localization

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Flag to track if warmup has been done
_MODEL_WARMED_UP = False

# Model configuration
PROJECT_ROOT = Path(__file__).parent

# Resolve model path at runtime (pick newest .keras under models/current/densenet121
# if the legacy expected filename doesn't exist). This mirrors the resolution used
# by the validation scripts so the web app won't fail when training produced
# timestamped filenames.
EXPECTED_MODEL_PATH = PROJECT_ROOT / "models/current/densenet121_finetuned.keras"

def resolve_model_path(expected_path: Path) -> Path:
    if expected_path.exists():
        return expected_path
    model_dir = expected_path.parent / 'densenet121'
    if model_dir.exists() and model_dir.is_dir():
        candidates = sorted(model_dir.glob('*.keras'), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    other_dir = expected_path.parent / 'resnet50'
    if other_dir.exists() and other_dir.is_dir():
        candidates = sorted(other_dir.glob('*.keras'), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return expected_path

MODEL_PATH = resolve_model_path(EXPECTED_MODEL_PATH)
MODEL_TYPE = "densenet121"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    
    # Warmup model on first prediction to avoid slow initial inference
    global _MODEL_WARMED_UP
    if not _MODEL_WARMED_UP:
        try:
            print("🔥 Warming up model (first prediction)...")
            warmup_start = time.time()
            
            # Create a dummy 128x128 RGB image
            dummy_img = np.random.rand(128, 128, 3) * 255
            dummy_path = app.config['UPLOAD_FOLDER'] / '_warmup_dummy.png'
            from PIL import Image as PILImage
            PILImage.fromarray(dummy_img.astype('uint8')).save(dummy_path)
            
            # Run a dummy prediction to compile graph and initialize GPU
            try:
                _ = predict_with_localization(
                    image_path=str(dummy_path),
                    model_path=MODEL_PATH,
                    model_type=MODEL_TYPE,
                    threshold_percentile=90,
                    confidence_threshold=0.80
                )
            finally:
                # Clean up dummy files
                if dummy_path.exists():
                    dummy_path.unlink()
                # Also clean up any generated visualization
                viz_path = app.config['UPLOAD_FOLDER'].parent / 'outputs' / 'predictions' / '_warmup_dummy_gradcam.png'
                if viz_path.exists():
                    viz_path.unlink()
            
            warmup_time = time.time() - warmup_start
            print(f"✅ Model warmed up in {warmup_time:.2f}s (subsequent predictions will be faster)")
            _MODEL_WARMED_UP = True
        except Exception as e:
            print(f"⚠️  Warmup failed (non-fatal): {e}")
            # Don't block the actual prediction even if warmup fails
            _MODEL_WARMED_UP = True
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, etc.)'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        
        # Run prediction
        results = predict_with_localization(
            image_path=str(filepath),
            model_path=MODEL_PATH,
            model_type=MODEL_TYPE,
            threshold_percentile=90,
            confidence_threshold=0.80  # 80% minimum confidence (stricter for safety)
        )
        
        # Read the visualization image and convert to base64
        with open(results['visualization_path'], 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up uploaded file
        filepath.unlink()
        
        # Convert all values to native Python types (fix JSON serialization)
        # Return results
        return jsonify({
            'success': True,
            'prediction': str(results['predicted_class']),
            'confidence': float(results['confidence']),
            'is_uncertain': bool(results['is_uncertain']),
            'entropy': float(results['entropy']),
            'confidence_threshold': float(results['confidence_threshold']),
            'warning_message': results['warning_message'],
            'probabilities': {k: float(v) for k, v in results['probabilities'].items()},
            'highlighted_area': float(results['activated_area_percent']),
            'visualization': img_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/about')
def about():
    """About page with model information."""
    return render_template('about.html')


@app.route('/metrics')
def metrics():
    """Metrics dashboard page."""
    return render_template('metrics.html')


@app.route('/api/metrics')
def api_metrics():
    """API endpoint for model and dataset metrics."""
    try:
        import pandas as pd
        
        # Check GPU without importing full TensorFlow
        gpu_available = False
        try:
            import tensorflow as tf
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        except:
            pass
        
        metrics_data = {
            'model': {
                'name': MODEL_TYPE,
                'path': str(MODEL_PATH.name),
                'architecture': 'DenseNet121 (Transfer Learning)',
                'input_shape': '128x128x3 (RGB)',
                'status': 'loaded' if MODEL_PATH.exists() else 'not_found'
            },
            'dataset': {
                'train_samples': 0,
                'test_samples': 0,
                'total_samples': 0,
                'classes': {},
                'sources': {}
            },
            'performance': {
                'reported_accuracy': '99.21%',
                'avg_inference_time': '~51 ms',
                'gpu_accelerated': gpu_available
            }
        }
        
        # Load dataset statistics
        train_csv = PROJECT_ROOT / "data/combined_data_splits/train_split.csv"
        test_csv = PROJECT_ROOT / "data/combined_data_splits/test_split.csv"
        
        if train_csv.exists():
            train_df = pd.read_csv(train_csv)
            metrics_data['dataset']['train_samples'] = len(train_df)
            
        if test_csv.exists():
            test_df = pd.read_csv(test_csv)
            metrics_data['dataset']['test_samples'] = len(test_df)
            
        if train_csv.exists() and test_csv.exists():
            combined_df = pd.concat([train_df, test_df])
            metrics_data['dataset']['total_samples'] = len(combined_df)
            
            # Class distribution
            class_counts = combined_df['class_name'].value_counts().to_dict()
            metrics_data['dataset']['classes'] = class_counts
            
            # Source distribution
            if 'source' in combined_df.columns:
                source_counts = combined_df['source'].value_counts().to_dict()
                metrics_data['dataset']['sources'] = source_counts
        
        # Add model metadata without loading the full model (too slow/memory intensive)
        if MODEL_PATH.exists():
            metrics_data['model']['total_parameters'] = 7_319_107  # DenseNet121 params
            metrics_data['model']['file_size_mb'] = round(MODEL_PATH.stat().st_size / (1024*1024), 2)
        
        return jsonify(metrics_data)
        
    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(f"Error in /api/metrics: {error_details}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*80)
    print("🧠 BRAIN TUMOR CLASSIFIER - WEB INTERFACE")
    print("="*80)
    # Try to report model accuracy from training_summary.csv (if present)
    reported_accuracy = None
    try:
        summary_csv = PROJECT_ROOT / 'models' / 'current' / 'training_summary.csv'
        if summary_csv.exists():
            import pandas as _pd
            df = _pd.read_csv(summary_csv)
            matched = df[df['model'].str.lower() == 'densenet121'] if 'model' in df.columns else _pd.DataFrame()
            if not matched.empty:
                reported_accuracy = float(matched.iloc[-1]['accuracy'])
    except Exception:
        reported_accuracy = None

    print(f"\n📁 Project Root: {PROJECT_ROOT}")
    print(f"🤖 Model path: {MODEL_PATH}")
    if reported_accuracy is not None:
        print(f"🤖 Reported accuracy (training_summary.csv): {reported_accuracy*100:.2f}%")
    else:
        # Keep actual hard-coded text as fallback
        print(f"🤖 Model: {MODEL_TYPE} (99.21% test accuracy on combined dataset)")
    print(f"📂 Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print("\n🌐 Starting server...")
    print("   Access at: http://localhost:5000")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
