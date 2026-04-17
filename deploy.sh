#!/bin/bash
set -euo pipefail

REPO_DIR="/var/www/LizardMorph"
LOG_FILE="/var/log/lizardmorph/deploy.log"
FRONTEND_DIR="$REPO_DIR/frontend"
NODE_BIN="/home/yloh30/.nvm/versions/node/v20.19.5/bin"
UV_BIN="/home/yloh30/.local/bin"

export PATH="$UV_BIN:$NODE_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

cd "$REPO_DIR"

log "========== Deployment started =========="

# Stash uncommitted changes if any
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    log "Stashing uncommitted changes..."
    git stash push -m "auto-deploy-stash-$(date +%s)"
fi

log "Fetching and resetting to origin/main..."
git fetch origin
git reset --hard origin/main
log "Git pull complete: $(git log --oneline -1)"

# Rebuild frontend
if [ -f "$FRONTEND_DIR/package.json" ]; then
    log "Installing frontend dependencies..."
    cd "$FRONTEND_DIR"
    npm ci --prefer-offline 2>>"$LOG_FILE" || npm install 2>>"$LOG_FILE"
    log "Building frontend..."
    npm run build 2>>"$LOG_FILE"
    log "Frontend build complete"
    cd "$REPO_DIR"
fi

# Restart backend via systemd
log "Restarting lizardmorph-backend service..."
sudo /bin/systemctl restart lizardmorph-backend

# Reload nginx to pick up any new static files
log "Reloading nginx..."
sudo /usr/sbin/nginx -s reload 2>>"$LOG_FILE" || true

log "========== Deployment finished =========="
