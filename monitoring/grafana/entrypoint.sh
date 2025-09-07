#!/bin/bash

echo "Custom entrypoint script starting..."

# Load environment variables from .env.config
if [ -f "/etc/grafana/.env.config" ]; then
    echo "Loading environment variables from .env.config"
    export $(cat /etc/grafana/.env.config | grep -v '^#' | xargs)
    echo "PROMETHEUS_USERNAME=$PROMETHEUS_USERNAME"
    echo "PROMETHEUS_PASSWORD=$PROMETHEUS_PASSWORD"
fi

# Replace environment variables in datasource configuration using sed
if [ -f "/etc/grafana/provisioning/datasources/prometheus.yml" ]; then
    echo "Processing datasource configuration..."
    # Create the processed file in a writable location
    sed "s/\${PROMETHEUS_USERNAME}/$PROMETHEUS_USERNAME/g; s/\${PROMETHEUS_PASSWORD}/$PROMETHEUS_PASSWORD/g" /etc/grafana/provisioning/datasources/prometheus.yml > /etc/grafana/provisioning/datasources/prometheus_processed.yml
    echo "Datasource configuration processed"
fi

echo "Starting Grafana..."
# Start Grafana
exec /run.sh "$@"
