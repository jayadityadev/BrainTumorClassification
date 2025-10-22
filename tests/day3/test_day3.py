"""
Day 3 Completion Test Script

This script validates that all Day 3 deliverables are present and correct.
Run this after completing all Day 3 notebooks to verify everything is in place.

Usage:
    python tests/test_day3_completion.py
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_test(name, passed, message=""):
    """Print test result"""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"  {status} - {name}")
    if message:
        print(f"         {Colors.YELLOW}{message}{Colors.END}")

def test_notebooks_exist():
    """Test 1: Check if all Day 3 notebooks exist"""
    print_header("Test 1: Notebooks Existence")
    
    notebooks = [
        'notebooks/day3/day3_01_data_splitting.ipynb',
        'notebooks/day3/day3_02_data_augmentation.ipynb',
        'notebooks/day3/day3_03_cnn_architecture.ipynb',
        'notebooks/day3/day3_04_training_test.ipynb'
    ]
    
    all_passed = True
    for notebook in notebooks:
        path = project_root / notebook
        exists = path.exists()
        print_test(notebook, exists, "" if exists else "File not found")
        all_passed = all_passed and exists
    
    return all_passed

def test_python_modules_exist():
    """Test 2: Check if Python modules exist"""
    print_header("Test 2: Python Modules")
    
    modules = [
        'src/modeling/data_generator.py',
        'src/modeling/model_cnn.py'
    ]
    
    all_passed = True
    for module in modules:
        path = project_root / module
        exists = path.exists()
        print_test(module, exists, "" if exists else "File not found")
        all_passed = all_passed and exists
    
    return all_passed

def test_data_splits_exist():
    """Test 3: Check if data split CSVs exist and are valid"""
    print_header("Test 3: Data Split Files")
    
    splits = ['train_split.csv', 'val_split.csv', 'test_split.csv']
    all_passed = True
    
    for split in splits:
        path = project_root / 'outputs' / 'data_splits' / split
        exists = path.exists()
        
        if exists:
            try:
                df = pd.read_csv(path)
                required_cols = ['filename', 'label', 'patient_id', 'filepath']
                has_cols = all(col in df.columns for col in required_cols)
                
                if has_cols:
                    print_test(f"{split} ({len(df)} rows)", True, 
                              f"Columns: {', '.join(df.columns.tolist())}")
                else:
                    print_test(split, False, "Missing required columns")
                    all_passed = False
            except Exception as e:
                print_test(split, False, f"Error reading file: {e}")
                all_passed = False
        else:
            print_test(split, False, "File not found")
            all_passed = False
    
    return all_passed

def test_patient_leakage():
    """Test 4: Check for patient leakage between splits"""
    print_header("Test 4: Patient Leakage Check")
    
    try:
        train_df = pd.read_csv(project_root / 'outputs' / 'data_splits' / 'train_split.csv')
        val_df = pd.read_csv(project_root / 'outputs' / 'data_splits' / 'val_split.csv')
        test_df = pd.read_csv(project_root / 'outputs' / 'data_splits' / 'test_split.csv')
        
        train_patients = set(train_df['patient_id'])
        val_patients = set(val_df['patient_id'])
        test_patients = set(test_df['patient_id'])
        
        train_val_overlap = train_patients & val_patients
        train_test_overlap = train_patients & test_patients
        val_test_overlap = val_patients & test_patients
        
        passed_train_val = len(train_val_overlap) == 0
        passed_train_test = len(train_test_overlap) == 0
        passed_val_test = len(val_test_overlap) == 0
        
        print_test("Train-Val separation", passed_train_val, 
                   "" if passed_train_val else f"{len(train_val_overlap)} patients overlap!")
        print_test("Train-Test separation", passed_train_test,
                   "" if passed_train_test else f"{len(train_test_overlap)} patients overlap!")
        print_test("Val-Test separation", passed_val_test,
                   "" if passed_val_test else f"{len(val_test_overlap)} patients overlap!")
        
        return passed_train_val and passed_train_test and passed_val_test
    
    except Exception as e:
        print_test("Patient leakage check", False, f"Error: {e}")
        return False

def test_config_files():
    """Test 5: Check if configuration files exist"""
    print_header("Test 5: Configuration Files")
    
    configs = [
        'outputs/configs/augmentation_config.json',
        'outputs/configs/model_architecture.json',
        'outputs/configs/model_summary.txt',
        'outputs/data_splits/split_summary.csv'
    ]
    
    all_passed = True
    for config in configs:
        path = project_root / config
        exists = path.exists()
        print_test(config, exists, "" if exists else "File not found")
        all_passed = all_passed and exists
    
    return all_passed

def test_training_history():
    """Test 6: Check if training history exists and shows learning"""
    print_header("Test 6: Training History")
    
    history_path = project_root / 'outputs' / 'configs' / 'day3_test_training_history.json'
    
    if not history_path.exists():
        print_test("Training history file", False, "File not found")
        return False
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Check structure
        has_history = 'history' in history
        print_test("History structure", has_history)
        
        if has_history:
            # Check if accuracy improved
            accuracies = history['history'].get('accuracy', [])
            val_accuracies = history['history'].get('val_accuracy', [])
            
            if len(accuracies) >= 2:
                improved = accuracies[-1] > accuracies[0]
                print_test("Training accuracy improved", improved,
                          f"{accuracies[0]:.3f} → {accuracies[-1]:.3f}")
            
            if len(val_accuracies) >= 1:
                better_than_random = val_accuracies[-1] > 0.333
                print_test("Validation > random (33.3%)", better_than_random,
                          f"Final: {val_accuracies[-1]*100:.1f}%")
                return better_than_random
        
        return has_history
    
    except Exception as e:
        print_test("Training history validation", False, f"Error: {e}")
        return False

def test_visualizations():
    """Test 7: Check if all visualizations were created"""
    print_header("Test 7: Visualizations")
    
    expected_viz = [
        'day3_01_data_distribution.png',
        'day3_01_split_distribution.png',
        'day3_02_augmentation_examples.png',
        'day3_02_training_batch.png',
        'day3_02_batch_distribution.png',
        'day3_02_all_classes_augmentation.png',
        'day3_03_model_architecture_custom.png',
        'day3_03_architecture_comparison.png',
        'day3_04_training_curves.png',
        'day3_04_sample_predictions.png'
    ]
    
    viz_dir = project_root / 'outputs' / 'visualizations'
    all_passed = True
    
    for viz in expected_viz:
        path = viz_dir / viz
        exists = path.exists()
        print_test(viz, exists, "" if exists else "Not found")
        all_passed = all_passed and exists
    
    return all_passed

def test_python_modules_importable():
    """Test 8: Check if Python modules can be imported"""
    print_header("Test 8: Module Imports")
    
    all_passed = True
    
    # Test data_generator
    try:
        from src.modeling import data_generator
        print_test("Import src.modeling.data_generator", True)
        
        # Check for key functions
        has_train_gen = hasattr(data_generator, 'create_train_generator')
        has_val_gen = hasattr(data_generator, 'create_val_test_generator')
        print_test("  - create_train_generator()", has_train_gen)
        print_test("  - create_val_test_generator()", has_val_gen)
        all_passed = all_passed and has_train_gen and has_val_gen
    except Exception as e:
        print_test("Import src.modeling.data_generator", False, str(e))
        all_passed = False
    
    # Test model_cnn
    try:
        from src.modeling import model_cnn
        print_test("Import src.modeling.model_cnn", True)
        
        # Check for key functions
        has_build = hasattr(model_cnn, 'build_cnn_model')
        has_print = hasattr(model_cnn, 'print_model_info')
        print_test("  - build_cnn_model()", has_build)
        print_test("  - print_model_info()", has_print)
        all_passed = all_passed and has_build and has_print
    except Exception as e:
        print_test("Import src.modeling.model_cnn", False, str(e))
        all_passed = False
    
    return all_passed

def test_class_distribution():
    """Test 9: Check class distribution across splits"""
    print_header("Test 9: Class Distribution Balance")
    
    try:
        train_df = pd.read_csv(project_root / 'outputs' / 'data_splits' / 'train_split.csv')
        val_df = pd.read_csv(project_root / 'outputs' / 'data_splits' / 'val_split.csv')
        test_df = pd.read_csv(project_root / 'outputs' / 'data_splits' / 'test_split.csv')
        
        all_passed = True
        
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if 'label' in df.columns:
                # Convert to string for counting
                df['label'] = df['label'].astype(str)
                dist = df['label'].value_counts(normalize=True) * 100
                
                # Check if all classes present
                has_all_classes = len(dist) == 3
                print_test(f"{name} has all 3 classes", has_all_classes,
                          f"Classes: {', '.join(dist.index.tolist())}")
                all_passed = all_passed and has_all_classes
        
        return all_passed
    
    except Exception as e:
        print_test("Class distribution check", False, f"Error: {e}")
        return False

def print_summary(results):
    """Print final summary"""
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"  Total Tests: {total}")
    print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
    print(f"  {Colors.RED}Failed: {failed}{Colors.END}")
    
    if failed == 0:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! Day 3 Complete!{Colors.END}\n")
        return True
    else:
        print(f"\n  {Colors.RED}{Colors.BOLD}⚠️  Some tests failed. Please review above.{Colors.END}\n")
        return False

def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}Day 3 Completion Test Suite{Colors.END}")
    print(f"{Colors.BOLD}Brain Tumor Classification Project{Colors.END}")
    
    results = {}
    
    # Run all tests
    results['Notebooks'] = test_notebooks_exist()
    results['Python Modules'] = test_python_modules_exist()
    results['Data Splits'] = test_data_splits_exist()
    results['No Patient Leakage'] = test_patient_leakage()
    results['Configuration Files'] = test_config_files()
    results['Training History'] = test_training_history()
    results['Visualizations'] = test_visualizations()
    results['Module Imports'] = test_python_modules_importable()
    results['Class Distribution'] = test_class_distribution()
    
    # Print summary
    all_passed = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
