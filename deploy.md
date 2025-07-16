# LizardMorph Deployment Guide

## Server Information
- **Host Location**: `/var/www/LizardMorph`
- **Port**: 80 (HTTP)
- **Web Server**: Gunicorn with Flask backend

## Prerequisites
Before deploying, ensure you have:
- Git access to the repository
- Node.js and npm installed (for frontend)
- Python and conda installed (for backend)

## Deployment Steps

### 1. Update Code
```bash
# Navigate to the project directory
cd /var/www/LizardMorph

# Pull latest changes from main branch
git pull origin main
```

### 2. Update Frontend Dependencies and Build
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Build for production
npm run build
```

### 3. Update Backend Dependencies (if needed)
```bash
# Activate conda environment
conda activate lizard

# Navigate to backend directory
cd ../backend

# Install/update Python dependencies using conda
conda install --file requirements.txt
```

### 4. Restart the Application
```bash
# Stop existing gunicorn processes (if running)
sudo pkill gunicorn

# Start the application with gunicorn
sudo $(which gunicorn) \
  --bind 0.0.0.0:80 \
  --daemon \
  --workers 1 \
  --timeout 120 \
  --access-logfile /var/log/lizardmorph/access.log \
  --error-logfile /var/log/lizardmorph/error.log \
  app:app
```

## Quick Deployment Script
For faster deployments, you can run this one-liner:
```bash
cd /var/www/LizardMorph && \
git pull origin main && \
cd frontend && npm install && npm run build && \
cd ../backend && conda activate lizard && conda install --file requirements.txt && \
sudo pkill gunicorn && \
sudo $(which gunicorn) --bind 0.0.0.0:80 --daemon --workers 1 app:app
```

## Monitoring and Logs
- **Access Logs**: `/var/log/lizardmorph/access.log`
- **Error Logs**: `/var/log/lizardmorph/error.log`
- **Process Status**: `ps aux | grep gunicorn`


### Useful Commands
```bash
# Check if gunicorn is running
ps aux | grep gunicorn

# View recent logs
tail -f /var/log/lizardmorph/error.log

