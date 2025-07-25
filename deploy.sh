#!/bin/bash

# Deploy script to keep /var/www/LizardMorph up to date with main branch

set -e  # Exit on any error

cd /var/www/LizardMorph

# Stash any local changes to avoid conflicts
git stash

# Switch to main branch and pull latest changes
git checkout main
git pull origin main

# If you want to stay on server branch but merge main into it:
# git checkout server
# git merge main

# Rebuild frontend if needed
if [ -d "frontend" ]; then
    cd frontend
    npm install
    npm run build
    cd ..
fi

# Restart backend services
sudo systemctl restart lizardmorph-backend || true

# Reload nginx
sudo nginx -s reload

echo "Deployment completed successfully!" 