# Multi-stage Dockerfile for React Frontend + Flask Backend

# ---- Frontend Build Stage ----
    FROM node:18 AS frontend-build

    # Set working directory for frontend
    WORKDIR /frontend-build
    
    # Copy frontend package files and install dependencies
    COPY frontend/package*.json ./
    RUN npm install
    
    # Copy frontend source code
    COPY frontend/ ./
    
    # Build the React application
    RUN npm run build
    
    # ---- Backend Stage ----
    # Using a specialized image with OpenCV pre-installed
    FROM python:3.8-slim
    
    # Set working directory
    WORKDIR /app
    
    # Install system dependencies for OpenCV, dlib and other libraries
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
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libcanberra-gtk* \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        gfortran \
        git \
        wget \
        vim \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy requirements file from backend directory
    COPY backend/requirements.txt .

    # First, install numpy which is required by OpenCV and dlib
    RUN pip3 install --upgrade pip && \
        pip3 install numpy==1.20.3
    
    # Install OpenCV with headless option (no GUI needed)
    RUN pip3 install opencv-python-headless==4.5.5.64
    
    # Install specific versions of dependencies separately
    RUN pip3 install pandas==1.3.5 && \
        pip3 install matplotlib==3.5.3 && \
        pip3 install Pillow==9.2.0 && \
        pip3 install Flask==2.2.5 && \
        pip3 install flask-cors==3.0.10 && \
        pip3 install gunicorn==20.1.0 && \
        pip3 install pydicom==2.3.1 && \
        pip3 install python-dotenv==1.0.0
    
    # Install dlib - this is often the trickiest one
    RUN pip3 install dlib==19.22.1
    
    # Create necessary directories
    RUN mkdir -p sessions
    # Define volume for backend code - this allows mounting at runtime
    VOLUME ["/app"]
    
    # Copy the backend code
    COPY backend/ .
    
    COPY --from=frontend-build /frontend-build/dist ./dist/

    # Explicitly copy favicon to both locations (root and static)
    COPY --from=frontend-build /frontend-build/dist/favicon.ico ./dist/favicon.ico
    COPY --from=frontend-build /frontend-build/dist/favicon.ico ./favicon.ico



    # Copy built frontend assets from the frontend build stage
    COPY --from=frontend-build /frontend-build/dist ./dist/

    # Expose port for Flask application
    EXPOSE 3000
    
    # Set environment variables
    ENV FLASK_APP=app.py
    ENV PYTHONUNBUFFERED=1
    ENV API_PORT=3000
    ENV FRONTEND_DIR="./dist"
    ENV PREDICTOR_FILE="/mnt/better_predictor_auto.dat"
    ENV SESSION_DIR="/mnt/sessions"
    # Use this for debugging Python imports
    #CMD ["python", "-c", "import sys; print(sys.path); import cv2; print('OpenCV loaded successfully'); import dlib; print('dlib loaded successfully')"]
    
    # Run with gunicorn pointing directly to app.py
    CMD ["gunicorn", "--bind", "0.0.0.0:3000", "--timeout", "600", "app:app"]