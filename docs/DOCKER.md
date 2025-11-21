# Docker usage for BrainTumorClassification

This document explains how to build and run the CPU-focused Docker image and how to use `docker-compose` for local development.

Build the image (from repo root):

```bash
docker build -t brain-tumor-app:latest .
```

Run the container (mount existing models/uploads/outputs to avoid baking them into the image):

```bash
docker run -it --rm -p 5000:5000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/uploads:/app/uploads" \
  -v "$PWD/outputs:/app/outputs" \
  brain-tumor-app:latest
```

Or use `docker-compose` (recommended for development):

```bash
docker compose up --build
```

Notes
- The image is CPU-focused and installs Python packages listed in `requirements-cpu.txt`.
- Large datasets and trained models are intentionally excluded from the image; use the mounted volumes above to provide them at runtime.
- To reproduce the entire pipeline (download data, preprocess, train), run the repository scripts inside a container or on the host. For large-scale training with GPU, create a separate GPU-enabled image (using NVIDIA base images) and run on a host with compatible drivers.
- If you need GPU support, tell me your target machine (Ubuntu + NVIDIA drivers, cloud, etc.) and I can add a `Dockerfile.gpu` and `docker-compose.gpu.yml` with instructions.
