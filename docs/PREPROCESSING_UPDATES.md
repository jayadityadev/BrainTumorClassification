# Preprocessing Pipeline Updates

## Overview
Enhanced the MAT to PNG conversion pipeline to extract additional metadata and tumor masks from the MATLAB files.

## What Changed

### 1. `src/convert_mat_to_png.py`
**New Features:**
- ✅ Extracts **Patient ID (PID)** from `cjdata.PID` field
- ✅ Extracts **Tumor Mask** from `cjdata.tumorMask` field  
- ✅ Encodes PID in output filenames: `pid{PID}_{original_name}.png`
- ✅ Saves masks separately in `outputs/ce_mri_masks/` (organized by label)
- ✅ Masks saved with same filename as corresponding images for easy pairing

**Output Structure:**
```
outputs/
├── ce_mri_images/          # Brain MRI images
│   ├── 0/                  # Label 0 (e.g., no tumor)
│   │   ├── pid1_1.png
│   │   ├── pid1_2.png
│   │   └── ...
│   ├── 1/                  # Label 1 (e.g., tumor)
│   └── ...
├── ce_mri_masks/           # Tumor masks (0=background, 255=tumor)
│   ├── 0/
│   ├── 1/
│   └── ...
└── logs/
    └── convert_errors.txt
```

### 2. `notebooks/day1_metadata.ipynb`
**New Features:**
- ✅ Extracts PID from filenames using regex: `pid(\d+)_(.+)\.png`
- ✅ Generates `metadata.csv` with columns:
  - `filename`: PNG filename
  - `label`: Class label (folder name)
  - `patient_id`: Extracted PID (for train/val/test splitting)
  - `original_mat_name`: Original .mat filename stem

**Why PID matters:**
- Prevents data leakage: same patient's scans shouldn't appear in both train and test sets
- Enables patient-level cross-validation
- Tracks multiple scans from the same patient

### 3. `notebooks/day1_visual_check.ipynb`
**New Features:**
- ✅ **Cell 1**: Quick view of one sample per class (image only)
- ✅ **Cell 2**: Side-by-side comparison showing:
  - Left: Original brain MRI image
  - Right: Image with red tumor mask overlay (alpha=0.4)
- ✅ Gracefully handles missing masks (displays "No mask found")

## MATLAB File Structure (for reference)
```python
cjdata:
  - PID          (6, 1)    uint16   # Patient ID
  - image        (512, 512) int16   # Brain MRI scan
  - label        (1, 1)    float64  # Class label (0/1)
  - tumorBorder  (1, 38)   float64  # Tumor boundary coordinates
  - tumorMask    (512, 512) uint8   # Binary mask (0=bg, 1=tumor)
```

## How to Use

### Step 1: Run Conversion (reprocess with new features)
```bash
python3 src/convert_mat_to_png.py
```
This will:
- Extract images to `outputs/ce_mri_images/`
- Extract masks to `outputs/ce_mri_masks/`
- Log errors to `outputs/logs/convert_errors.txt`

### Step 2: Generate Metadata
Run the first cell in `notebooks/day1_metadata.ipynb` to create:
- `outputs/ce_mri_images/metadata.csv` with PID and label info

### Step 3: Visualize Results
Run cells in `notebooks/day1_visual_check.ipynb` to:
- View sample images from each class
- See tumor mask overlays (red = tumor region)

## Benefits
1. **Better Train/Test Splitting**: Use PID to ensure patient-level splits
2. **Mask-Based Analysis**: Can now compute tumor size, location statistics
3. **Segmentation Ready**: Masks available if you want to train segmentation models
4. **Traceability**: Can trace PNG files back to original .mat files via filename

## Notes
- Masks are scaled 0→0, 1→255 for PNG visibility
- Uses `cv2.INTER_NEAREST` for mask resizing to preserve binary values
- PID extraction is robust: handles files with/without PID encoding
