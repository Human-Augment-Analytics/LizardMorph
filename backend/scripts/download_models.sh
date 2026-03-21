#!/bin/bash

# Configuration
REMOTE_HOST="login-ice.pace.gatech.edu"
REMOTE_PATH="/storage/ice-shared/cs8903onl/Lizard_Toepads/models/"
LOCAL_PATH="./models/"

echo "Downloading models from ${REMOTE_HOST}:${REMOTE_PATH} to ${LOCAL_PATH}..."

# Create local models directory if it doesn't exist
mkdir -p "${LOCAL_PATH}"

# Use rsync to download models
# -a: archive mode
# -v: verbose
# -z: compress during transfer
# --progress: show progress
rsync -avz --progress "${REMOTE_HOST}:${REMOTE_PATH}" "${LOCAL_PATH}"

if [ $? -eq 0 ]; then
    echo "Model download successful."
else
    echo "Model download failed. Please check your SSH connection and permissions."
    exit 1
fi
