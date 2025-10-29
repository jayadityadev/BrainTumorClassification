#!/usr/bin/env python3
"""
Dataset Download Script for Brain Tumor Classification Project
Downloads the required datasets from Figshare and Kaggle.
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import zipfile
import shutil
import requests
from tqdm import tqdm

# Get project root directory
project_root = Path(__file__).parent.parent


def download_file(url, destination):
    """Download a file with progress bar."""
    print(f"📥 Downloading from {url}")
    print(f"   Destination: {destination}")
    
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r   Progress: |{bar}| {percent:.1f}%', end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, destination, progress)
        print()  # New line after progress bar
        return True
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"📦 Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Error extracting: {e}")
        return False


def download_ce_mri_dataset():
    """
    Download the CE-MRI Brain Tumor Dataset from Figshare.
    
    The dataset comes as a nested ZIP structure:
    - Main ZIP contains 4 sub-ZIPs + cvind.mat + README.txt
    - Each sub-ZIP contains ~500-800 .mat files
    - Total: 3064 .mat files
    """
    dataset_dir = os.path.join(project_root, "datasets", "ce-mri")
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("📥 Downloading CE-MRI Brain Tumor Dataset from Figshare")
    print("="*70)
    print("\nℹ️  This dataset has a nested ZIP structure:")
    print("   - Main ZIP (1512427.zip) → 4 sub-ZIPs")
    print("   - Each sub-ZIP → 500-800 .mat files")
    print("   - Total: 3064 .mat files (~900MB)\n")
    
    # Download URL
    url = "https://figshare.com/ndownloader/articles/1512427/versions/5"
    main_zip_path = os.path.join(dataset_dir, "1512427.zip")
    temp_extract_dir = os.path.join(dataset_dir, "temp_extract")
    
    try:
        # Download main ZIP
        print(f"Downloading from: {url}")
        print("(Using browser-like headers to bypass bot protection...)")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.reason}")
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(main_zip_path, 'wb') as f, tqdm(
            desc="Downloading main ZIP",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        print("✓ Download successful!")
        
        # Extract main ZIP to temp directory
        print("\n📂 Extracting main ZIP...")
        os.makedirs(temp_extract_dir, exist_ok=True)
        with zipfile.ZipFile(main_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        print("✓ Main ZIP extracted!")
        
        # The ZIP contains files directly at root level (no subdirectory)
        # Files: 4 sub-ZIPs + cvind.mat + README.txt
        main_folder = temp_extract_dir
        print(f"   Looking for sub-ZIPs in: {main_folder}")
        
        # Extract all sub-ZIPs
        print("\n📦 Extracting sub-ZIPs...")
        sub_zips = [f for f in os.listdir(main_folder) if f.endswith('.zip')]
        
        mat_count = 0
        for sub_zip in tqdm(sub_zips, desc="Extracting sub-ZIPs"):
            sub_zip_path = os.path.join(main_folder, sub_zip)
            with zipfile.ZipFile(sub_zip_path, 'r') as zip_ref:
                # Extract .mat files directly to dataset_dir
                for member in zip_ref.namelist():
                    if member.endswith('.mat'):
                        # Extract to dataset_dir, preserving just the filename
                        filename = os.path.basename(member)
                        source = zip_ref.open(member)
                        target = open(os.path.join(dataset_dir, filename), "wb")
                        with source, target:
                            target.write(source.read())
                        mat_count += 1
        
        # Copy cvind.mat and README.txt
        for file in ['cvind.mat', 'README.txt']:
            src = os.path.join(main_folder, file)
            if os.path.exists(src):
                dst = os.path.join(dataset_dir, file)
                os.system(f'cp "{src}" "{dst}"')
        
        # Cleanup
        print("\n🧹 Cleaning up temporary files...")
        os.remove(main_zip_path)
        os.system(f'rm -rf "{temp_extract_dir}"')
        
        print(f"\n✅ SUCCESS!")
        print(f"   ✓ Extracted {mat_count} .mat files")
        print(f"   ✓ Dataset saved to: {dataset_dir}")
        return True
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n" + "="*70)
        print("📋 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("="*70)
        print("\n1. Open this URL in your browser:")
        print("   https://figshare.com/ndownloader/articles/1512427/versions/5")
        print("\n2. Download will start automatically as '1512427.zip' (~900MB)")
        print("\n3. Extract the ZIP file - you'll get a folder '1512427' containing:")
        print("   - brainTumorDataPublic_1-766.zip")
        print("   - brainTumorDataPublic_767-1532.zip")
        print("   - brainTumorDataPublic_1533-2298.zip")
        print("   - brainTumorDataPublic_2299-3064.zip")
        print("   - cvind.mat")
        print("   - README.txt")
        print("\n4. Extract each of the 4 sub-ZIP files")
        print("\n5. Copy all .mat files (3064 total) to:")
        print(f"   {dataset_dir}")
        print("\n6. Also copy cvind.mat and README.txt to the same location")
        print("\n" + "="*70)
        
        # Cleanup on failure
        if os.path.exists(main_zip_path):
            os.remove(main_zip_path)
        if os.path.exists(temp_extract_dir):
            os.system(f'rm -rf "{temp_extract_dir}"')
        
        return False
def download_kaggle_dataset(project_root):
    """Download Kaggle dataset using kaggle CLI."""
    print("\n" + "="*60)
    print("📚 Dataset 2: Kaggle Brain Tumor MRI")
    print("="*60)
    
    # Check if kaggle is installed
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Kaggle CLI not found!")
        print("\n📦 Install with: pip install kaggle")
        print("🔑 Then configure Kaggle API:")
        print("   1. Go to https://www.kaggle.com/settings")
        print("   2. Click 'Create New API Token'")
        print("   3. Place kaggle.json in ~/.kaggle/")
        return False
    
    datasets_dir = project_root / "datasets"
    kaggle_dir = datasets_dir / "kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading via Kaggle CLI...")
    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "masoudnickparvar/brain-tumor-mri-dataset",
            "-p", str(kaggle_dir)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        return False
    
    # Extract
    zip_file = kaggle_dir / "brain-tumor-mri-dataset.zip"
    if zip_file.exists():
        if not extract_zip(zip_file, kaggle_dir):
            return False
        zip_file.unlink()
    
    print("✓ Kaggle dataset ready!")
    return True


def main():
    """Main function."""
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║          📥 Brain Tumor Dataset Download Script 📥             ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

This script will download two datasets:

1. CE-MRI Dataset (~900MB)
   - Source: Figshare
   - Format: MATLAB .mat files (nested ZIP structure)
   - Images: 3,064 brain MRI scans
   - URL: https://figshare.com/ndownloader/articles/1512427/versions/5
   - Structure: Main ZIP → 4 sub-ZIPs → .mat files

2. Kaggle Brain Tumor Dataset (~150MB)
   - Source: Kaggle
   - Format: PNG images
   - Images: ~7,000 samples
   - Requires: Kaggle API credentials

ℹ️  Note: The Figshare download uses browser-like headers to avoid
   bot detection. If it fails, manual download instructions will be shown.

""")
    
    choice = input("Download both datasets? (y/n): ").lower()
    if choice != 'y':
        print("Exiting...")
        return
    
    # Download CE-MRI
    success_ce_mri = download_ce_mri_dataset()
    
    # Download Kaggle
    success_kaggle = download_kaggle_dataset(project_root)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Download Summary")
    print("="*60)
    print(f"CE-MRI Dataset:   {'✓ Success' if success_ce_mri else '✗ Failed (see manual instructions above)'}")
    print(f"Kaggle Dataset:   {'✓ Success' if success_kaggle else '✗ Failed'}")
    print("="*60)
    
    if success_ce_mri or success_kaggle:
        print("\n✅ Dataset download complete!")
        print("\n🔄 Next steps:")
        print("   1. Preprocess data: python src/preprocessing/convert_mat_to_png.py")
        print("   2. Enhance images: python src/preprocessing/enhance.py")
        print("   3. Train model: python src/models/fast_finetune_kaggle.py")
    
    if not success_kaggle:
        print("\n📥 Manual Download Instructions for Kaggle:")
        print("   1. Install Kaggle CLI: pip install kaggle")
        print("   2. Setup API: Get kaggle.json from https://www.kaggle.com/settings")
        print("   3. Place in: ~/.kaggle/kaggle.json")
        print("   4. Download: kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset")
        print("   5. Extract to: datasets/kaggle/")


if __name__ == "__main__":
    main()
