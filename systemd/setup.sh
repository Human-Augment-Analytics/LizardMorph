#!/bin/bash
# One-time setup for LizardMorph CI/CD pipeline
# Run with: sudo bash /var/www/LizardMorph/systemd/setup.sh
set -euo pipefail

REPO_DIR="/var/www/LizardMorph"

echo "=== LizardMorph CI/CD Setup ==="

# Ensure log directory exists
mkdir -p /var/log/lizardmorph
chown yloh30:www-data /var/log/lizardmorph
echo "[OK] Log directory"

# Install systemd service
cp "$REPO_DIR/systemd/lizardmorph-backend.service" /etc/systemd/system/
systemctl daemon-reload
echo "[OK] Systemd service installed"

# Install sudoers rule (validated before installing)
cp "$REPO_DIR/systemd/lizardmorph-deploy.sudoers" /etc/sudoers.d/lizardmorph-deploy
chmod 0440 /etc/sudoers.d/lizardmorph-deploy
if visudo -cf /etc/sudoers.d/lizardmorph-deploy; then
    echo "[OK] Sudoers rule installed and validated"
else
    echo "[FAIL] Invalid sudoers file — removing"
    rm -f /etc/sudoers.d/lizardmorph-deploy
    exit 1
fi

# Stop current manually-started backend (if running)
echo "Stopping any manually-started backend processes..."
pkill -f "uv run python app.py" 2>/dev/null || true
pkill -f "python app.py" 2>/dev/null || true
sleep 2

# Enable and start the service
systemctl enable lizardmorph-backend
systemctl start lizardmorph-backend
sleep 2

if systemctl is-active --quiet lizardmorph-backend; then
    echo "[OK] lizardmorph-backend service is running"
    systemctl status lizardmorph-backend --no-pager
else
    echo "[FAIL] Service did not start — check logs:"
    echo "  journalctl -u lizardmorph-backend -n 30 --no-pager"
    exit 1
fi

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Verify the app works: curl -s http://localhost:3000/api/health"
echo "  2. Add a GitHub webhook at:"
echo "     https://github.com/Human-Augment-Analytics/LizardMorph/settings/hooks/new"
echo "     Payload URL: https://haag-1.cc.gatech.edu/webhook"
echo "     Content type: application/json"
echo "     Secret: (use WEBHOOK_SECRET from .env)"
echo "     Events: Just the push event"
