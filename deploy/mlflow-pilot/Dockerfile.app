FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
ENV CMAKE_BUILD_PARALLEL_LEVEL=2
RUN pip install --no-cache-dir dlib==19.24.6
RUN pip install --no-cache-dir 'numpy==1.26.4' 'opencv-python-headless==4.11.0.86' 'onnxruntime==1.22.1' 'Flask==2.3.3' 'Werkzeug==2.3.8' 'flask-cors==4.0.0' 'pandas==2.2.3' 'Pillow==10.4.0' 'python-dotenv==1.0.1' 'psutil==6.1.1' 'prometheus-client==0.21.1' 'gunicorn==23.0.0' 'requests==2.32.3' 'pytest==8.3.5' 'pydicom==2.4.4' 'matplotlib==3.9.4'
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 ENABLE_ID_OCR=false PYTHONPATH=/source:/source/backend
COPY backend /source/backend
WORKDIR /work
CMD ["gunicorn", "--chdir", "/source/backend", "--bind", "0.0.0.0:3000", "--workers", "1", "--threads", "1", "--timeout", "600", "app:app"]
