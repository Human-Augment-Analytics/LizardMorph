# Dockerfile for LizardMorph MCP Server
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, dlib, and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    gfortran \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements/dependencies
COPY pyproject.toml ./
COPY README.md ./
COPY main.py ./
COPY utils.py ./
COPY better_predictor_auto.dat ./

# Install Python dependencies directly from pyproject.toml
RUN pip install --no-cache-dir \
    "aiofiles>=23.2.0" \
    "dlib>=19.24.0" \
    "mcp>=1.0.0" \
    "numpy>=1.24.0" \
    "opencv-python>=4.8.0" \
    "pillow>=10.0.0"

# Set environment variables
ENV PREDICTOR_FILE=/app/better_predictor_auto.dat

# Run the MCP server
CMD ["python", "main.py"]
