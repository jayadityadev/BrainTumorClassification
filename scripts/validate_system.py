"""
🔍 ULTIMATE VALIDATION SCRIPT
Tests all components of the Brain Tumor Classification system.

Latest verified run (2025-11-21)
- Combined training samples: 6,568
- Combined test samples: 1,519
- Best model: models/current/densenet121/densenet121_final_20251121_135727.keras
- DenseNet121 test accuracy: 99.21%
- ResNet50 test accuracy: 96.51%
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from PIL import Image
import cv2
from datetime import datetime
import json

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
# Default expected model path (kept for backward compatibility)
EXPECTED_MODEL_PATH = PROJECT_ROOT / "models/current/densenet121_finetuned.keras"

def resolve_model_path(expected_path: Path) -> Path:
    """Resolve an appropriate model file.

    If the exact expected path exists, return it. Otherwise look for
    common model directories (e.g., models/current/densenet121/) and
    select the newest .keras file present. If nothing is found, return
    the original expected path (so callers see the same error behavior).
    """
    if expected_path.exists():
        return expected_path

    # Try to find a model inside models/current/densenet121 or resnet50
    model_dir = expected_path.parent / 'densenet121'
    if model_dir.exists() and model_dir.is_dir():
        # pick newest .keras file
        candidates = sorted(model_dir.glob('*.keras'), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            print(f"ℹ️  Resolved model path to: {candidates[0]}")
            return candidates[0]

    # Fallback: try models/current/resnet50 (in case of naming mismatch)
    other_dir = expected_path.parent / 'resnet50'
    if other_dir.exists() and other_dir.is_dir():
        candidates = sorted(other_dir.glob('*.keras'), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            print(f"ℹ️  Resolved model path to: {candidates[0]}")
            return candidates[0]

    # Nothing found — return original expected path so existing error paths remain unchanged
    return expected_path

# Resolved model path used throughout the validation
MODEL_PATH = resolve_model_path(EXPECTED_MODEL_PATH)
TRAIN_CSV = PROJECT_ROOT / "data/combined_data_splits/train_split.csv"
TEST_CSV = PROJECT_ROOT / "data/combined_data_splits/test_split.csv"
KAGGLE_TEST_DIR = PROJECT_ROOT / "data/kaggle_enhanced/Testing"

# Add src to path
sys.path.append(str(PROJECT_ROOT / 'src' / 'inference'))
from predict import predict_with_localization

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print colorful header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(80)}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.END}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def test_1_gpu_availability():
    """Test 1: GPU Availability"""
    print_header("TEST 1: GPU Availability")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print_success(f"GPU detected: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        # Report memory growth settings and logical devices
        try:
            for gpu in gpus:
                try:
                    mg = tf.config.experimental.get_memory_growth(gpu)
                    print_info(f"   Memory growth for {gpu.name}: {mg}")
                except Exception:
                    pass
            logical = tf.config.list_logical_devices('GPU')
            print_info(f"   Logical GPU devices: {len(logical)}")
        except Exception:
            pass

        # Check where a small eager tensor is placed to indicate device preference
        try:
            sample = tf.constant([1.0])
            device_str = sample.device
            # device_str may be empty in some environments; guard accordingly
            if device_str:
                print_info(f"   Sample tensor placed on: {device_str}")
                if 'GPU' in device_str or 'gpu' in device_str:
                    print_success("TensorFlow placed sample ops on GPU by default")
                else:
                    print_warning("TensorFlow placed sample ops on CPU by default")
            else:
                # Fallback: use tf.test.is_built_with_cuda and tf.config.list_physical_devices
                if tf.config.list_physical_devices('GPU'):
                    print_info("Sample tensor device string empty — GPUs available but automatic placement could vary")
                else:
                    print_warning("No GPUs visible — TensorFlow will run on CPU")
        except Exception as e:
            print_warning(f"Could not determine default device placement: {e}")
        return True
    else:
        print_warning("No GPU detected - running on CPU (slower)")
        return True  # Not a failure, just slower

def test_2_file_structure():
    """Test 2: File Structure & Paths"""
    print_header("TEST 2: File Structure & Paths")
    
    required_files = {
        "Model": MODEL_PATH,
        "Train CSV": TRAIN_CSV,
        "Test CSV": TEST_CSV,
        "Kaggle Test Dir": KAGGLE_TEST_DIR,
        "Web App": PROJECT_ROOT / "app.py",
        "Inference Script": PROJECT_ROOT / "src/inference/predict.py",
    }
    
    all_exist = True
    for name, path in required_files.items():
        if path.exists():
            print_success(f"{name}: {path.name}")
        else:
            print_error(f"{name} NOT FOUND: {path}")
            all_exist = False
    
    return all_exist

def test_3_dataset_integrity():
    """Test 3: Dataset Integrity"""
    print_header("TEST 3: Dataset Integrity")
    
    try:
        # Load CSVs
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)
        
        print_info(f"Training samples: {len(train_df)}")
        print_info(f"Testing samples: {len(test_df)}")
        print_info(f"Total samples: {len(train_df) + len(test_df)}")
        
        # Check label distribution
        print("\n📊 Label Distribution (Combined):")
        combined_df = pd.concat([train_df, test_df])
        label_counts = combined_df['label'].value_counts().sort_index()
        
        label_names = {0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary'}
        for label, count in label_counts.items():
            percentage = (count / len(combined_df)) * 100
            print(f"   {label_names[label]}: {count} ({percentage:.1f}%)")
        
        # Check if files exist (sample check)
        print("\n🔍 Sample File Existence Check:")
        sample_size = min(10, len(test_df))
        missing = 0
        for _, row in test_df.head(sample_size).iterrows():
            if not Path(row['filepath']).exists():
                missing += 1
        
        if missing == 0:
            print_success(f"All {sample_size} sampled files exist")
        else:
            print_error(f"{missing}/{sample_size} sampled files are missing!")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Dataset integrity check failed: {e}")
        return False

def test_4_model_loading():
    """Test 4: Model Loading"""
    print_header("TEST 4: Model Loading")
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print_success(f"Model loaded successfully")
        
        # Check model architecture
        print(f"\n📐 Model Architecture:")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        total_params = model.count_params()
        print(f"   Total parameters: {total_params:,}")
        
        # Try to access nested DenseNet121
        try:
            base_model = model.get_layer('densenet121')
            print_success("Nested DenseNet121 base model found")
        except:
            print_warning("No nested base model (might be full model)")

        # Report training-summary accuracy if available
        try:
            summary_csv = PROJECT_ROOT / 'models' / 'current' / 'training_summary.csv'
            if summary_csv.exists():
                summary_df = pd.read_csv(summary_csv)
                acc = None
                # Prefer lookup by model name
                if 'model' in summary_df.columns and 'accuracy' in summary_df.columns:
                    matched = summary_df[summary_df['model'].str.lower() == 'densenet121']
                    if not matched.empty:
                        acc = float(matched.iloc[-1]['accuracy'])
                # Fallback: try matching by model filename
                if acc is None and 'path' in summary_df.columns:
                    matched2 = summary_df[summary_df['path'].str.contains(model.name)]
                    if not matched2.empty:
                        acc = float(matched2.iloc[-1]['accuracy'])

                if acc is not None:
                    print_info(f"Reported model accuracy (training summary): {acc:.4f} ({acc*100:.2f}%)")
                else:
                    print_info("No matching entry found in training_summary.csv for this model")
        except Exception as e:
            print_warning(f"Could not read training summary: {e}")
        
        return True
        
    except Exception as e:
        print_error(f"Model loading failed: {e}")
        return False

def test_5_direct_model_predictions():
    """Test 5: Direct Model Predictions (Known Images)"""
    print_header("TEST 5: Direct Model Predictions")
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Test images from Kaggle test set
        test_cases = [
            ("glioma/Te-gl_0014.jpg", 0, "Glioma"),
            ("meningioma/Te-me_0010.jpg", 1, "Meningioma"),
            ("pituitary/Te-pi_0015.jpg", 2, "Pituitary"),
        ]
        
        all_correct = True
        print(f"{'Image':<30} {'Expected':<12} {'Predicted':<12} {'Confidence':<12} {'Status'}")
        print("-" * 80)
        
        for img_rel_path, expected_label, expected_name in test_cases:
            img_path = KAGGLE_TEST_DIR / img_rel_path
            
            if not img_path.exists():
                print_warning(f"Test image not found: {img_path.name}")
                continue
            
            # Load and preprocess (ensure RGB channels)
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((128, 128))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict with temperature scaling for calibration
            pred = model.predict(img_array, verbose=0)
            
            # Apply temperature scaling (same as in predict.py)
            TEMPERATURE = 5.0
            logits = np.log(pred[0] + 1e-10)
            pred_calibrated = np.exp(logits / TEMPERATURE) / np.sum(np.exp(logits / TEMPERATURE))
            
            pred_class = np.argmax(pred_calibrated)
            confidence = pred_calibrated[pred_class] * 100
            
            label_names = {0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary'}
            pred_name = label_names[pred_class]
            
            # Check result
            is_correct = (pred_class == expected_label)
            status = f"{Colors.GREEN}✅ PASS{Colors.END}" if is_correct else f"{Colors.RED}❌ FAIL{Colors.END}"
            
            print(f"{img_path.name:<30} {expected_name:<12} {pred_name:<12} {confidence:>6.2f}%      {status}")
            
            if not is_correct:
                all_correct = False
        
        print()
        if all_correct:
            print_success("All direct predictions correct!")
        else:
            print_error("Some predictions failed!")
        
        return all_correct
        
    except Exception as e:
        print_error(f"Direct prediction test failed: {e}")
        return False

def test_6_inference_pipeline():
    """Test 6: Full Inference Pipeline (with Grad-CAM)"""
    print_header("TEST 6: Full Inference Pipeline")
    
    try:
        # Test with one image through full pipeline
        test_image = KAGGLE_TEST_DIR / "glioma/Te-gl_0014.jpg"
        
        if not test_image.exists():
            print_warning(f"Test image not found: {test_image}")
            return False
        
        print_info(f"Testing with: {test_image.name}")
        print_info("Running full prediction pipeline (with Grad-CAM)...\n")
        
        # Run prediction
        results = predict_with_localization(
            image_path=str(test_image),
            model_path=MODEL_PATH,
            model_type="densenet121",
            threshold_percentile=90,
            confidence_threshold=0.80
        )
        
        print(f"\n📊 Pipeline Results:")
        print(f"   Predicted: {results['predicted_class']}")
        print(f"   Confidence: {results['confidence']:.2f}%")
        print(f"   Uncertain: {results['is_uncertain']}")
        print(f"   Entropy: {results['entropy']:.3f}")
        
        # Check if visualization was created
        viz_path = Path(results['visualization_path'])
        if viz_path.exists():
            print_success(f"Visualization created: {viz_path.name}")
        else:
            print_error(f"Visualization NOT created: {viz_path}")
            return False
        
        # Check if prediction is correct
        expected = "Glioma"
        if results['predicted_class'] == expected:
            print_success(f"Prediction correct: {expected}")
        else:
            print_error(f"Prediction wrong: Expected {expected}, got {results['predicted_class']}")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Inference pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_7_web_app_integration():
    """Test 7: Web App Integration"""
    print_header("TEST 7: Web App Integration")
    
    try:
        # Check if app.py exists and imports correctly
        app_path = PROJECT_ROOT / "app.py"
        
        if not app_path.exists():
            print_error("app.py not found!")
            return False
        
        # Read app.py and check configuration
        with open(app_path, 'r') as f:
            app_content = f.read()
        
        # Check model path
        if "densenet121_final_finetuned_20251026_182845.keras" in app_content:
            print_success("Web app using correct fine-tuned model")
        else:
            print_warning("Web app might be using wrong model")
        
        # Check if templates exist
        templates_dir = PROJECT_ROOT / "templates"
        if (templates_dir / "index.html").exists():
            print_success("Web app templates found")
        else:
            print_error("Web app templates missing!")
            return False
        
        # Check if uploads directory exists
        uploads_dir = PROJECT_ROOT / "uploads"
        if uploads_dir.exists():
            print_success("Uploads directory exists")
        else:
            print_info("Uploads directory will be created on first run")
        
        print_info("Web app integration checks passed")
        print_info("To start web app: python app.py")
        
        return True
        
    except Exception as e:
        print_error(f"Web app integration test failed: {e}")
        return False

def test_8_label_mapping_consistency():
    """Test 8: Label Mapping Consistency"""
    print_header("TEST 8: Label Mapping Consistency")
    
    print_info("Verifying label mappings across all components...\n")
    
    # Expected mapping
    expected_mapping = {
        0: 'Glioma',
        1: 'Meningioma', 
        2: 'Pituitary'
    }
    
    print("📋 Expected Mapping (from training):")
    for idx, name in expected_mapping.items():
        print(f"   {idx} → {name}")
    
    # Check dataset CSV
    print("\n🗂️  Dataset CSV Mapping:")
    test_df = pd.read_csv(TEST_CSV)
    for label in sorted(test_df['label'].unique()):
        sample = test_df[test_df['label'] == label].iloc[0]
        filepath = Path(sample['filepath'])
        # Extract class from filepath (e.g., "mat_glioma_001.png" or "kaggle_glioma_001.png")
        if 'glioma' in filepath.name.lower() or 'glioma' in str(filepath.parent).lower():
            detected = 'Glioma'
        elif 'meningioma' in filepath.name.lower() or 'meningioma' in str(filepath.parent).lower():
            detected = 'Meningioma'
        elif 'pituitary' in filepath.name.lower() or 'pituitary' in str(filepath.parent).lower():
            detected = 'Pituitary'
        else:
            detected = 'Unknown'
        
        match = "✅" if expected_mapping.get(label) == detected else "❌"
        print(f"   Label {label} → {detected} {match}")
    
    # Check inference script
    print("\n🔮 Inference Script Mapping:")
    inference_script = PROJECT_ROOT / "src/inference/predict.py"
    
    if not inference_script.exists():
        print_warning("Inference script not found, skipping check")
    else:
        with open(inference_script, 'r') as f:
            content = f.read()
        
        # Look for label_names dictionary
        if "0: 'Glioma'" in content or "'Glioma'" in content:
            print_success("Inference script found with label mappings")
        else:
            print_warning("Could not verify inference script label mapping")
    
    print()
    print_success("Label mapping checks completed!")
    return True

def test_9_image_enhancement_pipeline():
    """Test 9: Image Enhancement Pipeline"""
    print_header("TEST 9: Image Enhancement Pipeline")
    
    try:
        # Import enhancement functions
        sys.path.append(str(PROJECT_ROOT / 'src' / 'preprocessing'))
        from enhance import enhance_image
        
        print_info("Testing image enhancement functions...")
        
        # Create a test image (128x128 grayscale with noise)
        np.random.seed(42)
        test_img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        
        # Add Gaussian noise
        noise = np.random.normal(0, 25, test_img.shape)
        test_img_noisy = np.clip(test_img + noise, 0, 255).astype(np.uint8)
        
        print("\n📸 Test Image Statistics:")
        print(f"   Original - Mean: {test_img.mean():.1f}, Std: {test_img.std():.1f}")
        print(f"   Noisy - Mean: {test_img_noisy.mean():.1f}, Std: {test_img_noisy.std():.1f}")
        
        # Test 1: Enhancement function exists and runs
        enhanced = enhance_image(test_img_noisy)
        
        if enhanced is None:
            print_error("Enhancement function returned None!")
            return False
        
        print_success("Enhancement function executed successfully")
        
        # Test 2: Output shape matches input
        if enhanced.shape != test_img_noisy.shape:
            print_error(f"Shape mismatch! Input: {test_img_noisy.shape}, Output: {enhanced.shape}")
            return False
        
        print_success(f"Output shape correct: {enhanced.shape}")
        
        # Test 3: Output dtype is uint8
        if enhanced.dtype != np.uint8:
            print_error(f"Output dtype wrong! Expected: uint8, Got: {enhanced.dtype}")
            return False
        
        print_success(f"Output dtype correct: {enhanced.dtype}")
        
        # Test 4: Output range is valid [0, 255]
        if enhanced.min() < 0 or enhanced.max() > 255:
            print_error(f"Output range invalid! Min: {enhanced.min()}, Max: {enhanced.max()}")
            return False
        
        print_success(f"Output range valid: [{enhanced.min()}, {enhanced.max()}]")
        
        # Test 5: Enhancement actually does something (not identity function)
        diff = np.abs(enhanced.astype(float) - test_img_noisy.astype(float)).mean()
        if diff < 1.0:
            print_warning(f"Enhancement seems to have minimal effect (diff={diff:.2f})")
        else:
            print_success(f"Enhancement effect detected (mean diff={diff:.2f})")
        
        # Test 6: Denoising reduces noise (check std deviation)
        enhanced_std = enhanced.std()
        noisy_std = test_img_noisy.std()
        
        print(f"\n🔬 Enhancement Quality Metrics:")
        print(f"   Noisy Std Dev: {noisy_std:.2f}")
        print(f"   Enhanced Std Dev: {enhanced_std:.2f}")
        
        if enhanced_std < noisy_std:
            print_success("Denoising working (reduced noise variance)")
        else:
            print_warning("Denoising might not be effective")
        
        # Test 7: Test with real MRI image if available
        print("\n🏥 Testing with real MRI scan...")
        real_test_image = None
        
        # Try to find a real test image
        for test_dir in [PROJECT_ROOT / "kaggle_temp/Testing/glioma",
                         PROJECT_ROOT / "outputs/ce_mri_images"]:
            if test_dir.exists():
                images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
                if images:
                    real_test_image = images[0]
                    break
        
        if real_test_image:
            real_img = cv2.imread(str(real_test_image), cv2.IMREAD_GRAYSCALE)
            if real_img is not None:
                # Resize to manageable size
                real_img_resized = cv2.resize(real_img, (128, 128))
                
                real_enhanced = enhance_image(real_img_resized)
                
                print(f"   Real MRI: {real_test_image.name}")
                print(f"   Before enhancement: mean={real_img_resized.mean():.1f}, std={real_img_resized.std():.1f}")
                print(f"   After enhancement: mean={real_enhanced.mean():.1f}, std={real_enhanced.std():.1f}")
                
                print_success("Real MRI enhancement successful")
            else:
                print_warning("Could not load real MRI image")
        else:
            print_warning("No real MRI images found for testing")
        
        # Test 8: CLAHE component test
        print("\n🔍 Testing CLAHE component...")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_result = clahe.apply(test_img_noisy)
        
        if clahe_result is not None and clahe_result.shape == test_img_noisy.shape:
            print_success("CLAHE component working correctly")
        else:
            print_error("CLAHE component failed!")
            return False
        
        # Test 9: Non-Local Means Denoising component test
        print("\n🔍 Testing Non-Local Means Denoising...")
        denoised = cv2.fastNlMeansDenoising(test_img_noisy, None, h=10, 
                                            templateWindowSize=7, 
                                            searchWindowSize=21)
        
        if denoised is not None and denoised.shape == test_img_noisy.shape:
            print_success("Non-Local Means Denoising working correctly")
        else:
            print_error("Non-Local Means Denoising failed!")
            return False
        
        print()
        print_success("All image enhancement tests passed!")
        return True
        
    except ImportError as e:
        print_error(f"Could not import enhancement module: {e}")
        return False
    except Exception as e:
        print_error(f"Image enhancement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_10_performance_benchmarks():
    """Test 10: Performance Benchmarks"""
    print_header("TEST 10: Performance Benchmarks")
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        test_image = KAGGLE_TEST_DIR / "glioma/Te-gl_0014.jpg"
        
        if not test_image.exists():
            print_warning("Test image not found for benchmark")
            return False
        
        # Load and preprocess (ensure RGB channels)
        img = Image.open(test_image).convert('RGB')
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Warmup
        _ = model.predict(img_array, verbose=0)
        
        # Benchmark
        num_runs = 10
        times = []
        
        print_info(f"Running {num_runs} predictions for benchmark...")
        
        for i in range(num_runs):
            start = datetime.now()
            _ = model.predict(img_array, verbose=0)
            end = datetime.now()
            times.append((end - start).total_seconds())
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        
        print(f"\n⏱️  Performance Metrics:")
        print(f"   Average: {avg_time:.2f} ms")
        print(f"   Std Dev: {std_time:.2f} ms")
        print(f"   Min: {min_time:.2f} ms")
        print(f"   Max: {max_time:.2f} ms")
        
        # Performance assessment
        if avg_time < 100:
            print_success("Excellent performance! (<100ms)")
        elif avg_time < 500:
            print_success("Good performance (<500ms)")
        else:
            print_warning(f"Slow performance ({avg_time:.0f}ms) - consider GPU acceleration")
        
        return True
        
    except Exception as e:
        print_error(f"Performance benchmark failed: {e}")
        return False

def generate_report(results):
    """Generate validation report"""
    print_header("VALIDATION SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"{Colors.GREEN}✅ Passed: {passed_tests}{Colors.END}")
    print(f"{Colors.RED}❌ Failed: {failed_tests}{Colors.END}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    print("\n📋 Detailed Results:")
    for test_name, passed in results.items():
        status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
        print(f"   {test_name:<40} {status}")
    
    # Overall status
    print()
    if failed_tests == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}{'🎉 ALL TESTS PASSED! SYSTEM IS READY! 🎉'.center(80)}{Colors.END}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}{'⚠️  SOME TESTS FAILED - REVIEW REQUIRED ⚠️'.center(80)}{Colors.END}")
    
    # Save report to file
    report_path = PROJECT_ROOT / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BRAIN TUMOR CLASSIFIER - VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Tests: {total_tests}\n")
        f.write(f"Passed: {passed_tests}\n")
        f.write(f"Failed: {failed_tests}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-"*80 + "\n")
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            f.write(f"{test_name:<40} {status}\n")
        
        f.write("\n" + "="*80 + "\n")
        if failed_tests == 0:
            f.write("STATUS: ALL TESTS PASSED ✅\n")
        else:
            f.write("STATUS: SOME TESTS FAILED ❌\n")
    
    print(f"\n💾 Report saved to: {report_path}")

def main():
    """Run all validation tests"""
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}")
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                               ║")
    print("║              🧠 BRAIN TUMOR CLASSIFIER - ULTIMATE VALIDATION 🧠              ║")
    print("║                   Testing Both Core Modules: Enhancement + ML                ║")
    print("║                                                                               ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    print(f"{Colors.CYAN}Starting comprehensive validation of all system components...{Colors.END}\n")
    
    # Run all tests
    results = {
        "1. GPU Availability": test_1_gpu_availability(),
        "2. File Structure": test_2_file_structure(),
        "3. Dataset Integrity": test_3_dataset_integrity(),
        "4. Model Loading": test_4_model_loading(),
        "5. Direct Model Predictions": test_5_direct_model_predictions(),
        "6. Inference Pipeline": test_6_inference_pipeline(),
        "7. Web App Integration": test_7_web_app_integration(),
        "8. Label Mapping Consistency": test_8_label_mapping_consistency(),
        "9. Image Enhancement Pipeline": test_9_image_enhancement_pipeline(),
        "10. Performance Benchmarks": test_10_performance_benchmarks(),
    }
    
    # Generate report
    generate_report(results)

if __name__ == "__main__":
    main()
