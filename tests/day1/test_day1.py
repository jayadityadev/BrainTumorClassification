"""
Test Script for Day 1: Dataset Loading and PNG Conversion

This script verifies all components from Day 1 of the Brain Tumor Classification project:
1. Raw dataset integrity (MAT files)
2. Metadata creation and accuracy
3. PNG conversion completeness
4. File structure and organization
5. Patient ID extraction
6. Class distribution validation

Usage:
    python tests/test_day1.py

Expected Results:
    - All tests should pass (✅)
    - 3,064 MAT files verified
    - 3,064 PNG images created
    - 233 unique patients identified
    - 3 tumor classes present (Meningioma, Glioma, Pituitary)

Author: Brain Tumor Classification Project
Date: October 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import cv2

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print a formatted section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")

def print_success(text):
    """Print a success message."""
    print(f"   {Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    """Print an error message."""
    print(f"   {Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    """Print an info message."""
    print(f"   {Colors.YELLOW}ℹ️  {text}{Colors.END}")


class Day1Tester:
    """Comprehensive test suite for Day 1 components."""
    
    def __init__(self):
        """Initialize the tester with project paths."""
        self.project_root = Path.cwd()
        self.dataset_dir = self.project_root / 'dataset'
        self.outputs_dir = self.project_root / 'outputs'
        self.metadata_path = self.outputs_dir / 'metadata.csv'
        self.png_dir = self.outputs_dir / 'ce_mri_images'
        
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
    
    def run_test(self, test_name, test_func):
        """
        Run a single test and track results.
        
        Args:
            test_name (str): Name of the test
            test_func (callable): Test function to execute
        """
        self.total_tests += 1
        try:
            result = test_func()
            if result:
                self.passed_tests += 1
                print_success(f"{test_name}: PASSED")
            else:
                self.failed_tests += 1
                print_error(f"{test_name}: FAILED")
        except Exception as e:
            self.failed_tests += 1
            print_error(f"{test_name}: ERROR - {str(e)}")
    
    def test_dataset_directory(self):
        """Test if dataset directory exists and contains files."""
        if not self.dataset_dir.exists():
            print_error("Dataset directory not found")
            return False
        
        mat_files = list(self.dataset_dir.glob('*.mat'))
        if len(mat_files) != 3064:
            print_error(f"Expected 3064 MAT files, found {len(mat_files)}")
            return False
        
        print_info(f"Found {len(mat_files)} MAT files")
        return True
    
    def test_png_class_distribution(self):
        """Test PNG class distribution matches expected counts."""
        expected_counts = {
            '1': 708,   # Meningioma
            '2': 1426,  # Glioma
            '3': 930    # Pituitary
        }
        
        actual_counts = {}
        for label in ['1', '2', '3']:
            label_dir = self.png_dir / label
            if label_dir.exists():
                png_files = list(label_dir.glob('*.png'))
                actual_counts[label] = len(png_files)
            else:
                actual_counts[label] = 0
        
        print_info(f"Class 1 (Meningioma): {actual_counts['1']} images")
        print_info(f"Class 2 (Glioma): {actual_counts['2']} images")
        print_info(f"Class 3 (Pituitary): {actual_counts['3']} images")
        
        if actual_counts == expected_counts:
            print_info(f"Total: {sum(actual_counts.values())} images (matches README)")
            return True
        else:
            for label in ['1', '2', '3']:
                if actual_counts[label] != expected_counts[label]:
                    print_error(f"Class {label}: expected {expected_counts[label]}, got {actual_counts[label]}")
            return False
    
    def test_metadata_file(self):
        """Test metadata.csv exists and has correct structure."""
        if not self.metadata_path.exists():
            print_error("metadata.csv not found")
            return False
        
        try:
            metadata = pd.read_csv(self.metadata_path)
            
            # Check required columns
            required_cols = ['filename', 'label', 'patient_id', 'original_mat_name']
            missing_cols = [col for col in required_cols if col not in metadata.columns]
            
            if missing_cols:
                print_error(f"Missing columns: {missing_cols}")
                return False
            
            # Check row count
            if len(metadata) != 3064:
                print_error(f"Expected 3064 rows, found {len(metadata)}")
                return False
            
            print_info(f"Metadata has {len(metadata)} rows and correct columns")
            return True
        except Exception as e:
            print_error(f"Error reading metadata: {str(e)}")
            return False
    
    def test_patient_extraction(self):
        """Test patient ID extraction from metadata."""
        try:
            metadata = pd.read_csv(self.metadata_path)
            unique_patients = metadata['patient_id'].nunique()
            
            if unique_patients < 200:  # Expected around 233
                print_error(f"Too few unique patients: {unique_patients}")
                return False
            
            print_info(f"Found {unique_patients} unique patients")
            
            # Check that patient IDs are numeric and reasonable
            sample_patient = metadata['patient_id'].iloc[0]
            if not (isinstance(sample_patient, (int, np.integer)) or str(sample_patient).isdigit()):
                print_error(f"Patient ID is not numeric: {sample_patient}")
                return False
            
            print_info(f"Patient IDs are valid (example: {sample_patient})")
            return True
        except Exception as e:
            print_error(f"Error checking patient IDs: {str(e)}")
            return False
    
    def test_png_conversion(self):
        """Test PNG files were created correctly."""
        if not self.png_dir.exists():
            print_error("PNG directory not found")
            return False
        
        # Count PNG files
        png_files = list(self.png_dir.rglob('*.png'))
        if len(png_files) != 3064:
            print_error(f"Expected 3064 PNG files, found {len(png_files)}")
            return False
        
        print_info(f"Found {len(png_files)} PNG files")
        return True
    
    def test_png_readability(self):
        """Test that PNG files are readable and valid."""
        try:
            metadata = pd.read_csv(self.metadata_path)
            sample = metadata.sample(10, random_state=42)
            
            for _, row in sample.iterrows():
                png_path = self.png_dir / str(row['label']) / row['filename']
                
                if not png_path.exists():
                    print_error(f"PNG file not found: {png_path}")
                    return False
                
                # Try to read the image
                img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
                if img is None or img.shape[0] == 0:
                    print_error(f"Cannot read PNG file: {png_path}")
                    return False
            
            print_info("Sample PNG files are readable and valid")
            return True
        except Exception as e:
            print_error(f"Error checking PNG readability: {str(e)}")
            return False
    
    def test_directory_structure(self):
        """Test that directory structure is organized by label."""
        expected_subdirs = ['1', '2', '3']
        
        for subdir in expected_subdirs:
            subdir_path = self.png_dir / subdir
            if not subdir_path.exists():
                print_error(f"Missing subdirectory: {subdir}")
                return False
            
            # Check that directory contains images
            images = list(subdir_path.glob('*.png'))
            if len(images) == 0:
                print_error(f"No images in subdirectory: {subdir}")
                return False
        
        print_info("Directory structure is correct (organized by class)")
        return True
    
    def test_metadata_consistency(self):
        """Test consistency between metadata and actual files."""
        try:
            metadata = pd.read_csv(self.metadata_path)
            
            # Check a sample of files
            sample = metadata.sample(20, random_state=42)
            
            for _, row in sample.iterrows():
                png_path = self.png_dir / str(row['label']) / row['filename']
                
                if not png_path.exists():
                    print_error(f"File in metadata not found: {png_path}")
                    return False
            
            print_info("Metadata is consistent with actual files")
            return True
        except Exception as e:
            print_error(f"Error checking metadata consistency: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all Day 1 tests."""
        print_header("🧪 DAY 1 TEST SUITE - Dataset Loading & PNG Conversion")
        
        print(f"\n{Colors.BOLD}Running tests...{Colors.END}\n")
        
        # Dataset tests
        print(f"\n{Colors.BOLD}1. Dataset Integrity Tests:{Colors.END}")
        self.run_test("Dataset directory exists", self.test_dataset_directory)
        self.run_test("PNG class distribution", self.test_png_class_distribution)
        
        # Metadata tests
        print(f"\n{Colors.BOLD}2. Metadata Tests:{Colors.END}")
        self.run_test("Metadata file structure", self.test_metadata_file)
        self.run_test("Patient ID extraction", self.test_patient_extraction)
        self.run_test("Metadata consistency", self.test_metadata_consistency)
        
        # PNG conversion tests
        print(f"\n{Colors.BOLD}3. PNG Conversion Tests:{Colors.END}")
        self.run_test("PNG files created", self.test_png_conversion)
        self.run_test("PNG files readable", self.test_png_readability)
        self.run_test("Directory structure", self.test_directory_structure)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary."""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}TEST SUMMARY:{Colors.END}")
        print(f"{Colors.BOLD}{'='*70}{Colors.END}")
        
        print(f"\n   Total tests: {self.total_tests}")
        print(f"   {Colors.GREEN}Passed: {self.passed_tests}{Colors.END}")
        print(f"   {Colors.RED}Failed: {self.failed_tests}{Colors.END}")
        
        if self.failed_tests == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ ALL DAY 1 TESTS PASSED!{Colors.END}")
            print(f"{Colors.GREEN}Dataset is ready for Day 2 processing.{Colors.END}\n")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ SOME TESTS FAILED!{Colors.END}")
            print(f"{Colors.RED}Please review the errors above.{Colors.END}\n")
            return 1


def main():
    """Main function to run Day 1 tests."""
    tester = Day1Tester()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
