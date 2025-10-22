"""
Test Script for Day 2: Image Enhancement

This script verifies all components from Day 2 of the Brain Tumor Classification project:
1. Enhanced image creation and completeness
2. Enhancement quality (contrast improvement)
3. Image integrity and readability
4. Directory structure for enhanced images
5. Consistency with original images
6. Pixel value validation

Usage:
    python tests/test_day2.py

Expected Results:
    - All tests should pass (✅)
    - 3,064 enhanced images created
    - Average contrast improvement > 30%
    - All images readable and valid
    - Proper directory organization by class

Author: Brain Tumor Classification Project
Date: October 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List

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


class Day2Tester:
    """Comprehensive test suite for Day 2 components."""
    
    def __init__(self):
        """Initialize the tester with project paths."""
        self.project_root = Path.cwd()
        self.outputs_dir = self.project_root / 'outputs'
        self.metadata_path = self.outputs_dir / 'metadata.csv'
        self.original_dir = self.outputs_dir / 'ce_mri_images'
        self.enhanced_dir = self.outputs_dir / 'ce_mri_enhanced'
        
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
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """
        Calculate image contrast using standard deviation.
        
        Args:
            image (np.ndarray): Input grayscale image
            
        Returns:
            float: Contrast value (standard deviation of pixel intensities)
        """
        return np.std(image)
    
    def calculate_improvement(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Calculate percentage improvement in contrast.
        
        Args:
            original (np.ndarray): Original image
            enhanced (np.ndarray): Enhanced image
            
        Returns:
            float: Percentage improvement
        """
        orig_contrast = self.calculate_contrast(original)
        enh_contrast = self.calculate_contrast(enhanced)
        
        if orig_contrast == 0:
            return 0.0
        
        improvement = ((enh_contrast - orig_contrast) / orig_contrast) * 100
        return improvement
    
    def test_enhanced_directory(self):
        """Test if enhanced images directory exists."""
        if not self.enhanced_dir.exists():
            print_error("Enhanced images directory not found")
            return False
        
        print_info(f"Enhanced directory found: {self.enhanced_dir}")
        return True
    
    def test_enhanced_count(self):
        """Test that all images have been enhanced."""
        enhanced_files = list(self.enhanced_dir.rglob('*.png'))
        
        if len(enhanced_files) != 3064:
            print_error(f"Expected 3064 enhanced images, found {len(enhanced_files)}")
            return False
        
        print_info(f"Found {len(enhanced_files)} enhanced images")
        return True
    
    def test_directory_structure(self):
        """Test that enhanced directory has correct structure."""
        expected_subdirs = ['1', '2', '3']
        
        for subdir in expected_subdirs:
            subdir_path = self.enhanced_dir / subdir
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
    
    def test_image_readability(self):
        """Test that enhanced images are readable."""
        try:
            metadata = pd.read_csv(self.metadata_path)
            sample = metadata.sample(10, random_state=42)
            
            for _, row in sample.iterrows():
                img_path = self.enhanced_dir / str(row['label']) / row['filename']
                
                if not img_path.exists():
                    print_error(f"Enhanced image not found: {img_path}")
                    return False
                
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None or img.shape[0] == 0:
                    print_error(f"Cannot read enhanced image: {img_path}")
                    return False
            
            print_info("Sample enhanced images are readable")
            return True
        except Exception as e:
            print_error(f"Error checking image readability: {str(e)}")
            return False
    
    def test_image_dimensions(self):
        """Test that enhanced images maintain original dimensions."""
        try:
            metadata = pd.read_csv(self.metadata_path)
            sample = metadata.sample(10, random_state=42)
            
            for _, row in sample.iterrows():
                orig_path = self.original_dir / str(row['label']) / row['filename']
                enh_path = self.enhanced_dir / str(row['label']) / row['filename']
                
                orig_img = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
                enh_img = cv2.imread(str(enh_path), cv2.IMREAD_GRAYSCALE)
                
                if orig_img.shape != enh_img.shape:
                    print_error(f"Dimension mismatch for {row['filename']}")
                    print_error(f"  Original: {orig_img.shape}, Enhanced: {enh_img.shape}")
                    return False
            
            print_info("Enhanced images maintain original dimensions")
            return True
        except Exception as e:
            print_error(f"Error checking dimensions: {str(e)}")
            return False
    
    def test_contrast_improvement(self):
        """Test that images have improved contrast."""
        try:
            metadata = pd.read_csv(self.metadata_path)
            sample = metadata.sample(20, random_state=42)
            
            improvements = []
            
            for _, row in sample.iterrows():
                orig_path = self.original_dir / str(row['label']) / row['filename']
                enh_path = self.enhanced_dir / str(row['label']) / row['filename']
                
                if not orig_path.exists() or not enh_path.exists():
                    continue
                
                orig_img = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
                enh_img = cv2.imread(str(enh_path), cv2.IMREAD_GRAYSCALE)
                
                improvement = self.calculate_improvement(orig_img, enh_img)
                improvements.append(improvement)
            
            avg_improvement = np.mean(improvements)
            min_improvement = np.min(improvements)
            max_improvement = np.max(improvements)
            
            print_info(f"Average contrast improvement: {avg_improvement:.1f}%")
            print_info(f"Range: {min_improvement:.1f}% to {max_improvement:.1f}%")
            
            # Check if improvement is significant (at least 30% on average)
            if avg_improvement < 30:
                print_error(f"Insufficient contrast improvement: {avg_improvement:.1f}%")
                return False
            
            return True
        except Exception as e:
            print_error(f"Error calculating contrast improvement: {str(e)}")
            return False
    
    def test_pixel_value_range(self):
        """Test that enhanced images have valid pixel value range."""
        try:
            metadata = pd.read_csv(self.metadata_path)
            sample = metadata.sample(10, random_state=42)
            
            for _, row in sample.iterrows():
                img_path = self.enhanced_dir / str(row['label']) / row['filename']
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                if img.min() < 0 or img.max() > 255:
                    print_error(f"Invalid pixel values in {row['filename']}")
                    print_error(f"  Range: [{img.min()}, {img.max()}]")
                    return False
            
            print_info("Enhanced images have valid pixel range [0, 255]")
            return True
        except Exception as e:
            print_error(f"Error checking pixel values: {str(e)}")
            return False
    
    def test_no_empty_images(self):
        """Test that no enhanced images are completely black or white."""
        try:
            metadata = pd.read_csv(self.metadata_path)
            sample = metadata.sample(20, random_state=42)
            
            for _, row in sample.iterrows():
                img_path = self.enhanced_dir / str(row['label']) / row['filename']
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                # Check if image is all black
                if np.max(img) == 0:
                    print_error(f"Image is completely black: {row['filename']}")
                    return False
                
                # Check if image is all white
                if np.min(img) == 255:
                    print_error(f"Image is completely white: {row['filename']}")
                    return False
            
            print_info("No empty (black/white) images detected")
            return True
        except Exception as e:
            print_error(f"Error checking empty images: {str(e)}")
            return False
    
    def test_file_consistency(self):
        """Test that all original images have corresponding enhanced versions."""
        try:
            metadata = pd.read_csv(self.metadata_path)
            
            missing_files = []
            for _, row in metadata.iterrows():
                orig_path = self.original_dir / str(row['label']) / row['filename']
                enh_path = self.enhanced_dir / str(row['label']) / row['filename']
                
                if orig_path.exists() and not enh_path.exists():
                    missing_files.append(row['filename'])
            
            if missing_files:
                print_error(f"Missing {len(missing_files)} enhanced images")
                print_error(f"  Examples: {missing_files[:3]}")
                return False
            
            print_info("All original images have enhanced versions")
            return True
        except Exception as e:
            print_error(f"Error checking file consistency: {str(e)}")
            return False
    
    def test_class_distribution(self):
        """Test that class distribution is maintained after enhancement."""
        try:
            class_counts = {}
            
            for class_label in [1, 2, 3]:
                class_dir = self.enhanced_dir / str(class_label)
                if class_dir.exists():
                    class_counts[class_label] = len(list(class_dir.glob('*.png')))
                else:
                    class_counts[class_label] = 0
            
            print_info(f"Class 1 (Meningioma): {class_counts[1]} images")
            print_info(f"Class 2 (Glioma): {class_counts[2]} images")
            print_info(f"Class 3 (Pituitary): {class_counts[3]} images")
            
            # Verify all classes have images
            if all(count > 0 for count in class_counts.values()):
                return True
            else:
                print_error("Some classes have no enhanced images")
                return False
        except Exception as e:
            print_error(f"Error checking class distribution: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all Day 2 tests."""
        print_header("🧪 DAY 2 TEST SUITE - Image Enhancement")
        
        print(f"\n{Colors.BOLD}Running tests...{Colors.END}\n")
        
        # Directory and structure tests
        print(f"\n{Colors.BOLD}1. Directory Structure Tests:{Colors.END}")
        self.run_test("Enhanced directory exists", self.test_enhanced_directory)
        self.run_test("Enhanced image count", self.test_enhanced_count)
        self.run_test("Directory structure", self.test_directory_structure)
        self.run_test("File consistency", self.test_file_consistency)
        
        # Image quality tests
        print(f"\n{Colors.BOLD}2. Image Quality Tests:{Colors.END}")
        self.run_test("Image readability", self.test_image_readability)
        self.run_test("Image dimensions", self.test_image_dimensions)
        self.run_test("Pixel value range", self.test_pixel_value_range)
        self.run_test("No empty images", self.test_no_empty_images)
        
        # Enhancement effectiveness tests
        print(f"\n{Colors.BOLD}3. Enhancement Quality Tests:{Colors.END}")
        self.run_test("Contrast improvement", self.test_contrast_improvement)
        self.run_test("Class distribution", self.test_class_distribution)
        
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
            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ ALL DAY 2 TESTS PASSED!{Colors.END}")
            print(f"{Colors.GREEN}Enhanced images are ready for Day 3 processing.{Colors.END}\n")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ SOME TESTS FAILED!{Colors.END}")
            print(f"{Colors.RED}Please review the errors above.{Colors.END}\n")
            return 1


def main():
    """Main function to run Day 2 tests."""
    tester = Day2Tester()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
