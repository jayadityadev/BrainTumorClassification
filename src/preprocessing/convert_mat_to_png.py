# src/convert_mat_to_png.py
import os
import glob
import scipy.io as sio
import numpy as np
import cv2
from tqdm import tqdm
import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # project root (two levels up from this file)
DATASET_DIR = os.path.join(ROOT, "datasets", "ce-mri")
OUT_DIR = os.path.join(ROOT, "data", "ce_mri_images")
LOG_DIR = os.path.join(ROOT, "outputs", "logs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def is_hdf5_mat(path):
    # MATLAB v7.3+ .mat files are HDF5 files with this 8-byte signature
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        return header == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False


def normalize_to_uint8(img):
    img = img.astype("float32")
    minv, maxv = img.min(), img.max()
    if maxv == minv:
        return (np.zeros_like(img) + 128).astype("uint8")
    img = (img - minv) / (maxv - minv)
    img = (img * 255.0).astype("uint8")
    return img


def extract_from_scipy_mat(mat):
    # Try common struct name 'cjdata'
    if "cjdata" in mat:
        cj = mat["cjdata"]
        # many times it's nested: cj['image'][0][0]
        try:
            image = cj["image"][0][0]
        except Exception:
            image = cj["image"].squeeze()
        # label often nested
        try:
            label = int(cj["label"][0][0][0][0])
        except Exception:
            try:
                label = int(np.squeeze(cj["label"]))
            except Exception:
                label = 0
        
        # Extract PID (Patient ID) - may be stored as ASCII character codes
        # PID can be numeric or alphanumeric
        pid = None
        if "PID" in cj:
            try:
                pid_val = cj["PID"][0][0]
                # Check if it's ASCII character codes
                if isinstance(pid_val, np.ndarray) and pid_val.dtype.kind in ('u', 'i') and pid_val.size > 1:
                    # Convert ASCII array to string
                    pid = ''.join(chr(int(x)) for x in pid_val.flatten())
                else:
                    pid = str(np.squeeze(pid_val))
            except Exception:
                pid = None
        
        # Extract tumor mask
        mask = None
        if "tumorMask" in cj:
            try:
                mask = cj["tumorMask"][0][0]
            except Exception:
                try:
                    mask = cj["tumorMask"].squeeze()
                except Exception:
                    mask = None
    else:
        # fallback: try to find an array that looks like a grayscale image
        candidates = {k: v for k, v in mat.items() if not k.startswith("__")}
        best = None
        best_size = 0
        for k, v in candidates.items():
            if isinstance(v, np.ndarray):
                arr = v
                # squeeze singletons
                if arr.ndim > 2 and 1 in arr.shape:
                    arr = np.squeeze(arr)
                if arr.ndim == 2:
                    size = arr.shape[0] * arr.shape[1]
                    if size > best_size:
                        best = v
                        best_size = size
        if best is None:
            raise ValueError("No image-like 2D array found in scipy mat dict.")
        image = best
        label = 0
        pid = None
        mask = None
    return image, label, pid, mask


def extract_from_h5(path):
    """
    Open an HDF5 (.mat v7.3) file and try to find the best candidate datasets
    for image (largest 2D numeric dataset) and label (dataset name containing 'label').
    Also extract PID and tumorMask if available.
    This attempts to dereference MATLAB-style references when needed.
    """
    with h5py.File(path, "r") as f:
        datasets = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, obj))

        f.visititems(visitor)

        # find label dataset first (by name)
        label = None
        for name, ds in datasets:
            if "label" in name.lower():
                try:
                    val = ds[()]
                    # dereference if it's an array of object/ref
                    if isinstance(val, np.ndarray):
                        val = np.squeeze(val)
                        if val.size == 1:
                            label = int(val)
                            break
                    else:
                        label = int(val)
                        break
                except Exception:
                    continue

        # Extract PID (stored as ASCII character codes in some files)
        # PID can be numeric (e.g., "100360") or alphanumeric (e.g., "MR0402480D")
        pid = None
        for name, ds in datasets:
            if "pid" in name.lower():
                try:
                    val = ds[()]
                    if isinstance(val, np.ndarray):
                        # Check if it's ASCII character codes
                        if val.dtype.kind in ('u', 'i') and val.size > 1:
                            # Convert ASCII array to string
                            pid = ''.join(chr(int(x)) for x in val.flatten())
                        else:
                            val = np.squeeze(val)
                            pid = str(val)
                    else:
                        pid = str(val)
                    break
                except Exception:
                    continue

        # Extract tumor mask
        mask = None
        for name, ds in datasets:
            if "tumormask" in name.lower():
                try:
                    arr = ds[...]
                    if isinstance(arr, np.ndarray):
                        arr = np.squeeze(arr)
                        if arr.ndim == 2:
                            mask = arr
                            break
                except Exception:
                    continue

        # find image-like dataset: prefer numeric, largest 2D after squeeze
        best = None
        best_size = 0
        for name, ds in datasets:
            arr = None
            try:
                # if dataset stores numeric data directly
                if np.issubdtype(ds.dtype, np.number):
                    arr = ds[...]
                else:
                    # attempt to dereference the first element (common for MATLAB cell refs)
                    data = ds[...]
                    if isinstance(data, np.ndarray) and data.size > 0:
                        first = data.flatten()[0]
                        # handle HDF5 object/reference
                        try:
                            arr = f[first][...]
                        except Exception:
                            # sometimes stored as bytes representing numeric arrays - skip
                            arr = None
                if arr is None:
                    continue
                # squeeze singleton dimensions
                if arr.ndim > 2 and 1 in arr.shape:
                    arr = np.squeeze(arr)
                if arr.ndim == 2:
                    size = int(arr.shape[0]) * int(arr.shape[1])
                    if size > best_size:
                        best = arr
                        best_size = size
            except Exception:
                continue

        if best is None:
            # As last resort, try any dataset that becomes 2D after squeeze
            for name, ds in datasets:
                try:
                    arr = ds[...]
                    arr = np.asarray(arr)
                    if arr.ndim > 2 and 1 in arr.shape:
                        arr = np.squeeze(arr)
                    if arr.ndim == 2:
                        best = arr
                        break
                except Exception:
                    continue

        if best is None:
            raise ValueError("No suitable image dataset found in HDF5 .mat file.")

        if label is None:
            label = 0

        return best, int(label), pid, mask


def get_image_and_label(mat_file):
    # Prefer opening with h5py first (covers MATLAB v7.3 HDF5 .mat files).
    # If that fails, fall back to scipy.loadmat (v7.2 and earlier). If
    # scipy raises an error (e.g. NotImplementedError for v7.3), try h5py
    # as a final fallback and re-raise the original exception if both fail.
    # Using h5py first avoids depending on a small header check that may
    # not always be reliable across all files.
    # Returns: image, label, pid, mask
    try:
        try:
            # Try to open with h5py to detect/handle v7.3 files.
            with h5py.File(mat_file, "r"):
                image, label, pid, mask = extract_from_h5(mat_file)
                return image, label, pid, mask
        except Exception:
            # Not an HDF5-backed .mat (or h5py couldn't open it) — try scipy
            mat = sio.loadmat(mat_file)
            image, label, pid, mask = extract_from_scipy_mat(mat)
            return image, label, pid, mask
    except Exception as load_err:
        # If scipy.loadmat raised NotImplementedError (v7.3) or any other
        # loading issue, attempt one last time with h5py before giving up.
        try:
            image, label, pid, mask = extract_from_h5(mat_file)
            return image, label, pid, mask
        except Exception:
            # Re-raise the original loading error to preserve the root cause
            raise load_err


def main(resize=(128, 128)):
    files = glob.glob(os.path.join(DATASET_DIR, "*.mat"))
    if len(files) == 0:
        print("No .mat files found in", DATASET_DIR)
        return

    # Also create a masks directory
    MASK_DIR = os.path.join(ROOT, "outputs", "ce_mri_masks")
    os.makedirs(MASK_DIR, exist_ok=True)

    for fpath in tqdm(files):
        try:
            image, label, pid, mask = get_image_and_label(fpath)

            # Normalize and resize image
            img_uint8 = normalize_to_uint8(np.asarray(image))
            if resize is not None:
                img_uint8 = cv2.resize(img_uint8, resize, interpolation=cv2.INTER_AREA)

            # Create filename with PID if available
            base_name = os.path.splitext(os.path.basename(fpath))[0]
            if pid is not None:
                fname = f"pid{pid}_{base_name}.png"
            else:
                fname = f"{base_name}.png"

            # Save image in label folder
            label_folder = os.path.join(OUT_DIR, str(label))
            os.makedirs(label_folder, exist_ok=True)
            out_path = os.path.join(label_folder, fname)
            cv2.imwrite(out_path, img_uint8)

            # Save mask if available
            if mask is not None:
                mask_uint8 = np.asarray(mask, dtype="uint8")
                if resize is not None:
                    mask_uint8 = cv2.resize(mask_uint8, resize, interpolation=cv2.INTER_NEAREST)
                
                mask_label_folder = os.path.join(MASK_DIR, str(label))
                os.makedirs(mask_label_folder, exist_ok=True)
                mask_path = os.path.join(mask_label_folder, fname)
                cv2.imwrite(mask_path, mask_uint8 * 255)  # scale 0/1 to 0/255 for visibility

        except Exception as e:
            # log and continue
            with open(os.path.join(LOG_DIR, "convert_errors.txt"), "a") as f:
                f.write(f"{fpath}\t{repr(e)}\n")


if __name__ == "__main__":
    main(resize=(128, 128))
