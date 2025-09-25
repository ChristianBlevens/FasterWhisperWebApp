FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Set environment variables for optimal GPU performance
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_LAUNCH_BLOCKING=0

# Install system dependencies (cuDNN already included in base image)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Verify cuDNN installation (already included in base image)
RUN python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'cuDNN version: {torch.backends.cudnn.version()}' if torch.cuda.is_available() else 'CUDA not available')" || echo "PyTorch not yet installed"

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Verify moviepy installation
RUN python3 -c "import moviepy.editor; print('MoviePy successfully installed')" || echo "MoviePy installation failed"

# Create necessary directories
RUN mkdir -p downloads transcripts static templates

# Copy application files
COPY . .


# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "app:app"]