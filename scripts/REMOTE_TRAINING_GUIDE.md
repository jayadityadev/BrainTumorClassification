# 🚀 Remote GPU Training Guide (RTX 5080)

## Quick Start: 3 Methods to Train on Roommate's GPU

---

## Method 1: SSH + Rsync (Recommended - Simplest)

### Step 1: Get Roommate's IP Address
On your roommate's machine:
```bash
hostname -I
# Or: ip a | grep 'inet ' | grep -v '127.0.0.1'
```
Example output: `192.168.1.100`

### Step 2: Test SSH Connection
```bash
# From your machine
ssh roommate@192.168.1.100
```

If password-less login doesn't work, set up SSH key:
```bash
ssh-copy-id roommate@192.168.1.100
```

### Step 3: Sync Project to Remote Machine
```bash
# From your machine
cd /projects/ai-ml/BrainTumorProject

rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'dataset/' \
  . roommate@192.168.1.100:~/BrainTumorProject/
```

### Step 4: SSH Into Remote Machine & Install Dependencies
```bash
ssh roommate@192.168.1.100

cd ~/BrainTumorProject

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow scipy

# Verify GPU is available
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Step 5: Run Training
```bash
# Still on remote machine
cd ~/BrainTumorProject

# Convert notebook to Python script (or use Jupyter)
jupyter nbconvert --to script notebooks/day4/day4_02_full_training.ipynb --output run_training.py

# Run training (detached session so you can close SSH)
nohup python3 run_training.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### Step 6: Copy Results Back
```bash
# From your machine (after training completes)
rsync -avz roommate@192.168.1.100:~/BrainTumorProject/outputs/ \
  /projects/ai-ml/BrainTumorProject/outputs/
```

---

## Method 2: VS Code Remote SSH (Better Developer Experience)

### Setup:
1. Install "Remote - SSH" extension in VS Code
2. Press `F1` → "Remote-SSH: Connect to Host"
3. Enter: `roommate@192.168.1.100`
4. Open folder: `~/BrainTumorProject`
5. Install Python extension on remote
6. Run notebooks directly on remote machine!

### Advantages:
- ✅ Edit files directly on remote machine
- ✅ Run notebooks in remote Jupyter kernel
- ✅ See real-time GPU usage
- ✅ No need to sync files back and forth

---

## Method 3: Jupyter Remote Server (For Notebooks)

### On Roommate's Machine:
```bash
cd ~/BrainTumorProject
source .venv/bin/activate

# Start Jupyter with network access
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

### On Your Machine:
```bash
# Forward port through SSH
ssh -N -L 8888:localhost:8888 roommate@192.168.1.100
```

Then open `http://localhost:8888` in your browser - you'll see roommate's Jupyter!

---

## Method 4: VS Code Live Share (Real-time Collaboration)

This is for **collaborative editing**, not remote execution:
- ⚠️ Code still runs on YOUR machine (GTX 1650)
- Only useful if you want to edit together
- NOT what you want for GPU training

---

## Recommended Workflow

**For this project, use Method 1 (SSH + Rsync):**

```bash
# 1. One-time setup on remote
ssh roommate@192.168.1.100
cd ~
git clone [or rsync from your machine]
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow scipy

# 2. Sync files whenever you make changes
rsync -avz --exclude '.venv' /projects/ai-ml/BrainTumorProject/ \
  roommate@192.168.1.100:~/BrainTumorProject/

# 3. Run training remotely
ssh roommate@192.168.1.100 'cd ~/BrainTumorProject && source .venv/bin/activate && python3 -m notebooks.day4.train'

# 4. Copy results back
rsync -avz roommate@192.168.1.100:~/BrainTumorProject/outputs/ ./outputs/
```

---

## Training Time Comparison

| Hardware | Training Time | Memory Usage |
|----------|--------------|--------------|
| Your GTX 1650 (4GB) | ❌ Out of Memory | N/A |
| Your CPU (Ryzen 5600H) | ~25 minutes | ~2GB RAM |
| Roommate's RTX 5080 (16GB) | **~3-5 minutes** ⚡ | ~3GB VRAM |

**The RTX 5080 will be 5-8x faster than CPU!**

---

## Quick Decision Guide

**Choose SSH + Rsync if:**
- ✅ You just want to train once and get results
- ✅ You're comfortable with command line
- ✅ Simple, fast setup

**Choose VS Code Remote-SSH if:**
- ✅ You'll be training multiple times
- ✅ You want to debug/iterate on remote GPU
- ✅ Better development experience

**Don't use Live Share for:**
- ❌ Remote GPU training (it doesn't do that)

---

## Need Help?

1. Can't SSH? Check if roommate's firewall allows connections
2. GPU not detected? Run `nvidia-smi` on remote machine
3. Out of memory? Shouldn't happen with 16GB, but reduce batch size to 16 if needed
