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

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src' / 'inference'))
from predict import predict_with_localization

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

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
        # Keep previous hard-coded text as fallback
        print(f"🤖 Model: {MODEL_TYPE} (99.23% accuracy on Kaggle, 97.47% combined)")
    print(f"📂 Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print("\n🌐 Starting server...")
    print("   Access at: http://localhost:5000")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
