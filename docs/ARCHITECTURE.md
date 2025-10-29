# System Architecture

## Overview

The Brain Tumor Classification System is a production-grade medical imaging AI application that combines image enhancement with deep learning classification.

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                         │
│                                                                   │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │  Web App     │         │ Command Line │                      │
│  │  (Flask)     │         │   Interface  │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
└─────────┼──────────────────────────┼───────────────────────────┘
          │                          │
          └──────────┬───────────────┘
                     │
┌────────────────────┼────────────────────────────────────────────┐
│                    │   APPLICATION LAYER                         │
│                    ▼                                             │
│         ┌───────────────────┐                                   │
│         │  Inference Engine │                                   │
│         │  (src/inference)  │                                   │
│         └─────────┬─────────┘                                   │
│                   │                                              │
│         ┌─────────┴─────────┐                                   │
│         │                   │                                   │
│    ┌────▼────┐      ┌──────▼──────┐                           │
│    │ Predict │      │   Grad-CAM  │                           │
│    │ Module  │      │ Visualization│                           │
│    └────┬────┘      └──────┬──────┘                           │
└─────────┼──────────────────┼─────────────────────────────────┘
          │                  │
┌─────────┼──────────────────┼─────────────────────────────────┐
│         │    MODEL LAYER   │                                   │
│         │                  │                                   │
│    ┌────▼──────────────────▼────┐                             │
│    │   DenseNet121 Model        │                             │
│    │   (models/current/)        │                             │
│    │                            │                             │
│    │   • 7.7M parameters        │                             │
│    │   • RGB input (128×128×3)  │                             │
│    │   • 3 classes output       │                             │
│    │   • 99.23% accuracy        │                             │
│    └────┬───────────────────────┘                             │
└─────────┼─────────────────────────────────────────────────────┘
          │
┌─────────┼─────────────────────────────────────────────────────┐
│         │    PREPROCESSING LAYER                               │
│         │                                                       │
│    ┌────▼─────────────────────┐                               │
│    │ Image Enhancement        │                               │
│    │ (src/preprocessing)      │                               │
│    │                          │                               │
│    │  1. Non-Local Means      │                               │
│    │     Denoising            │                               │
│    │  2. CLAHE Enhancement    │                               │
│    │  3. Normalization        │                               │
│    └────┬─────────────────────┘                               │
└─────────┼─────────────────────────────────────────────────────┘
          │
┌─────────▼─────────────────────────────────────────────────────┐
│                       DATA LAYER                               │
│                                                                 │
│  ┌─────────┐    ┌──────────┐    ┌─────────────┐              │
│  │  Raw    │    │Processed │    │   Splits    │              │
│  │  .mat   │───▶│  Images  │───▶│  Train/Test │              │
│  │  Files  │    │   (PNG)  │    │    (CSV)    │              │
│  └─────────┘    └──────────┘    └─────────────┘              │
│                                                                 │
│  Dataset: 11,349 images (3 tumor types)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Module Details

### 1. Data Layer (`data/`)

**Purpose**: Stores and organizes all datasets

**Structure**:
- `raw/`: Original .mat files from dataset
- `processed/`: Converted PNG images
- `splits/`: Train/test split CSVs

**Data Flow**:
```
.mat files → PNG conversion → Train/Test split → Model training
```

### 2. Preprocessing Layer (`src/preprocessing/`)

**Purpose**: Image enhancement pipeline

**Components**:
- **enhance.py**: Main enhancement module
  - Non-Local Means Denoising (h=10)
  - CLAHE (clipLimit=2.0, tileGridSize=8×8)
  - Normalization to [0, 255]
  
- **convert_mat_to_png.py**: Dataset conversion utility

**Performance**: ~50-100ms per image (CPU)

### 3. Model Layer (`models/`)

**Current Production Model**: 
- `models/current/densenet121_finetuned.keras`
- Architecture: DenseNet121 with custom classifier
- Input: 128×128×3 RGB images
- Output: 3 classes (Glioma, Meningioma, Pituitary)
- Parameters: 7,697,475

**Training Details**:
- Base: DenseNet121 (ImageNet pretrained)
- Method: Transfer learning + fine-tuning
- Dataset: Combined .mat + Kaggle (11,349 images)
- Accuracy: 97.47% combined, 99.23% Kaggle test

### 4. Inference Layer (`src/inference/`)

**predict.py**: Main prediction module
- Loads model
- Preprocesses input image
- Returns predictions with confidence
- Generates 3-panel visualization

**gradcam.py**: Explainability module
- Generates Grad-CAM heatmaps
- Highlights regions model focuses on
- Uses last convolutional layer activations

### 5. Application Layer

**app.py**: Flask web application
- Simple upload interface
- Real-time predictions
- Visualization display
- Runs on localhost:5000

**scripts/**: Utility scripts
- `validate_system.py`: Comprehensive system validation
- `evaluate_kaggle.py`: Model evaluation on test set

## Data Flow

### Training Pipeline
```
1. Raw Data (.mat files)
   ↓
2. Convert to PNG (convert_mat_to_png.py)
   ↓
3. Combine with Kaggle dataset (combine_datasets.py)
   ↓
4. Split into train/test (80/20)
   ↓
5. Apply augmentation during training
   ↓
6. Train DenseNet121 model
   ↓
7. Save to models/current/
```

### Inference Pipeline
```
1. User uploads MRI scan
   ↓
2. Load image (RGB)
   ↓
3. Resize to 128×128
   ↓
4. Normalize to [0, 1]
   ↓
5. Model prediction
   ↓
6. Generate Grad-CAM heatmap
   ↓
7. Create 3-panel visualization
   ↓
8. Return prediction + confidence
```

## Technology Stack

### Core Libraries
- **TensorFlow 2.20.0**: Deep learning framework
- **Keras**: High-level neural networks API
- **OpenCV (cv2)**: Image processing
- **NumPy**: Numerical computations
- **Pillow (PIL)**: Image loading/saving

### Web Framework
- **Flask**: Web application framework
- **Werkzeug**: WSGI utilities

### Visualization
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization

### Data Processing
- **Pandas**: Data manipulation
- **scikit-learn**: ML utilities and metrics

## Performance Characteristics

### Inference Speed
- **GPU (GTX 1650)**: 60-100ms per image
- **CPU**: 200-500ms per image

### Memory Usage
- Model: ~300MB loaded in memory
- GPU: 2.6GB VRAM available
- Peak RAM: ~1.5GB during inference

### Accuracy Metrics
- **Kaggle Test Set**: 99.23% (899/906 correct)
- **Combined Dataset**: 97.47%
- **Per-class Precision**: >98% for all classes

## Deployment Architecture

### Development Setup
```
Local Machine
├── Python 3.11 + Virtual Environment
├── GPU Support (optional but recommended)
└── Flask Development Server
```

### Production Recommendations
```
Production Server
├── Python 3.11
├── GPU Server (NVIDIA with CUDA)
├── Gunicorn/uWSGI (WSGI server)
├── Nginx (Reverse proxy)
└── Docker Container (optional)
```

## Security Considerations

1. **Input Validation**: Only accept image formats
2. **File Size Limits**: Max 16MB uploads
3. **Sanitization**: Secure filename handling
4. **Error Handling**: Graceful failure on invalid inputs
5. **Rate Limiting**: Prevent DoS attacks (recommended for production)

## Scalability

### Current Capacity
- Single request: <100ms
- Concurrent requests: Limited by single worker
- Daily throughput: ~10,000 predictions (single instance)

### Scaling Options
1. **Horizontal Scaling**: Multiple Flask workers
2. **Load Balancing**: Distribute requests across instances
3. **Batch Processing**: Process multiple images simultaneously
4. **Model Optimization**: TensorRT/ONNX conversion
5. **Caching**: Cache predictions for identical images

## Monitoring & Logging

### Current Logging
- Terminal output for predictions
- Validation reports in `outputs/reports/`
- Model training history saved as JSON

### Recommended Production Logging
- Request/response logging
- Error tracking (Sentry)
- Performance metrics (Prometheus)
- Health checks endpoint
- Model prediction logs with confidence scores

## Future Enhancements

1. **Multi-model Ensemble**: Combine multiple architectures
2. **Uncertainty Quantification**: Bayesian neural networks
3. **Active Learning**: Improve model with new data
4. **DICOM Support**: Handle medical imaging standard
5. **3D Volume Analysis**: Process full MRI scans
6. **API Authentication**: Token-based access control
7. **Database Integration**: Store predictions and feedback

## Directory Structure

```
BrainTumorProject/
├── app.py                    # Web application
├── README.md                 # Project overview
├── requirements.txt          # Dependencies
│
├── config/                   # Configuration files
├── src/                      # Source code
│   ├── data/                 # Data processing
│   ├── preprocessing/        # Image enhancement
│   ├── models/               # Model training
│   └── inference/            # Prediction engine
│
├── scripts/                  # Utility scripts
├── models/                   # Trained models
├── data/                     # Datasets
├── outputs/                  # Generated outputs
├── docs/                     # Documentation
├── notebooks/                # Archived experiments
└── templates/                # Web templates
```

## References

- DenseNet: [Huang et al., 2017](https://arxiv.org/abs/1608.06993)
- Grad-CAM: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
- Transfer Learning: [Yosinski et al., 2014](https://arxiv.org/abs/1411.1792)

---

**Last Updated**: October 26, 2025  
**System Version**: 1.0.0  
**Status**: Production Ready
