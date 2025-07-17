# Dockerfile for LizardMorph MCP Server with Web Interface
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
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gfortran \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    "mcp==1.0.0" \
    "numpy>=1.24.0" \
    "opencv-python-headless>=4.8.0" \
    "dlib>=19.24.0" \
    "pillow>=10.0.0"

# Copy application files
COPY main.py ./
COPY utils.py ./
COPY better_predictor_auto.dat ./

# Create outputs directory for web server
RUN mkdir -p /app/outputs

# Set environment variables
ENV PREDICTOR_FILE=/app/better_predictor_auto.dat
ENV WEB_PORT=8080
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app

USER mcpuser

# Expose web server port
EXPOSE 8080

# Run the MCP server
CMD ["python", "main.py"]