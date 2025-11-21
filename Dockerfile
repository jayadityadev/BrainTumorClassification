FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies commonly needed by ML/image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements-cpu.txt ./requirements-cpu.txt
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Copy only the repo files (large data are excluded by .dockerignore)
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
