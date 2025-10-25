"""
Generate Metadata CSV for Original Dataset

This script scans the extracted PNG images and creates a metadata CSV
with filename, label, patient_id, and original_mat_name.

Extracted from: notebooks/day1/day1_metadata.ipynb

Author: Jayaditya Dev
Date: October 24, 2025
"""

import csv
import glob
import os
import re
from pathlib import Path

def generate_metadata():
    """Generate metadata CSV from extracted PNG images."""
    
    project_root = Path(__file__).parent.parent.parent
    base = project_root / "outputs" / "ce_mri_images"
    rows = []
    
    print("🔍 Scanning images in:", base)
    
    for lbl in os.listdir(base):
        p = os.path.join(base, lbl)
        if not os.path.isdir(p): 
            continue
        
        for f in glob.glob(os.path.join(p, "*.png")):
            fname = os.path.basename(f)
            
            # Extract PID from filename (format: pid{PID}_originalname.png)
            # PID can be numeric (e.g., pid100360_1.png) or alphanumeric (e.g., pidMR0402480D_2376.png)
            pid_match = re.match(r'pid([^_]+)_(.+)\.png', fname)
            if pid_match:
                pid = pid_match.group(1)  # Keep as string (can be numeric or alphanumeric)
                orig_name = pid_match.group(2)
            else:
                pid = None
                orig_name = os.path.splitext(fname)[0]
            
            # Add filepath for compatibility with combined metadata
            filepath = str(Path(f).absolute())
            
            rows.append([fname, lbl, pid, orig_name, filepath])
    
    # Save metadata CSV to data_splits directory (consistent with combined metadata)
    output_dir = project_root / "outputs" / "data_splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = output_dir / "metadata.csv"
    
    with open(metadata_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "label", "patient_id", "original_mat_name", "filepath"])
        writer.writerows(rows)
    
    print(f"\n✅ Saved metadata to: {metadata_path}")
    print(f"   Total images: {len(rows)}")
    
    unique_patients = len(set(r[2] for r in rows if r[2] is not None))
    print(f"   Unique patients: {unique_patients}")
    
    images_without_pid = sum(1 for r in rows if r[2] is None)
    print(f"   Images without PID: {images_without_pid}")
    
    # Show class distribution
    class_counts = {}
    for row in rows:
        label = row[1]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"\n📊 Class Distribution:")
    class_names = {'1': 'Meningioma', '2': 'Glioma', '3': 'Pituitary'}
    for label in sorted(class_counts.keys()):
        count = class_counts[label]
        name = class_names.get(label, f'Class {label}')
        percentage = count / len(rows) * 100
        print(f"   Class {label} ({name}): {count} images ({percentage:.1f}%)")
    
    # Show a few sample rows
    if rows:
        print("\n📋 Sample entries:")
        for row in rows[:3]:
            print(f"   {row[0]} - Label: {row[1]}, Patient: {row[2]}")
    
    return metadata_path

if __name__ == "__main__":
    generate_metadata()
