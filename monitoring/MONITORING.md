# LizardMorph Monitoring with Grafana

This project now uses **Grafana** for comprehensive monitoring instead of Flask-MonitoringDashboard.

## 🚀 Quick Start

### 1. Start the Monitoring Stack
```bash
./setup_monitoring.sh
```

### 2. Access the Dashboards
- **Grafana Dashboard**: http://localhost:3001
- **Prometheus**: http://localhost:9090

## 📊 What's Monitored

### Application Metrics
- **Request Rate**: Requests per second
- **Response Time**: Average response time per endpoint
- **Total Requests**: Cumulative request count
- **Error Rate**: Failed requests

### System Metrics
- **CPU Usage**: Real-time CPU utilization
- **Memory Usage**: RAM usage percentage
- **Disk Usage**: Storage utilization

### Custom Metrics
- Session creation rate
- Image processing time
- File upload metrics

## 🔧 Configuration

### Prometheus Configuration
- **Scrape Interval**: 15s for general metrics, 5s for Flask app
- **Targets**: Flask app on port 3000
- **Metrics Endpoints**: `/metrics` and `/system/metrics`

### Grafana Dashboard
- **Auto-refresh**: 5 seconds
- **Time Range**: Last hour (configurable)
- **Alerts**: Configurable thresholds

## 📁 File Structure
```
monitoring/
├── prometheus.yml          # Prometheus configuration
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── prometheus.yml
│       └── dashboards/
│           ├── dashboard.yml
│           └── lizardmorph-dashboard.json
docker-compose.monitoring.yml
setup_monitoring.sh
```

## 🛠️ Management Commands

### Start Monitoring Stack
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### Stop Monitoring Stack
```bash
docker-compose -f docker-compose.monitoring.yml down
```

### View Logs
```bash
docker-compose -f docker-compose.monitoring.yml logs -f
```

### Restart Services
```bash
docker-compose -f docker-compose.monitoring.yml restart
```

## 🔍 Metrics Endpoints

### Flask App Metrics
- **URL**: `http://localhost:3000/metrics`
- **Format**: Prometheus text format
- **Content**: HTTP request metrics, custom counters

### System Metrics
- **URL**: `http://localhost:3000/system/metrics`
- **Format**: Prometheus text format
- **Content**: CPU, memory, disk usage

## 📈 Dashboard Features

### Real-time Monitoring
- Live system resource usage
- Request rate visualization
- Response time tracking
- Error rate monitoring

### Historical Data
- 200 hours of data retention
- Trend analysis
- Performance baselines

### Alerting (Configurable)
- CPU usage > 90%
- Memory usage > 90%
- High error rates
- Slow response times

## 🔐 Security

### Authentication
- Grafana: Username/password authentication
- Prometheus: No authentication (internal network)

### Network Access
- Grafana: Port 3001 (external access)
- Prometheus: Port 9090 (internal only)
- Flask App: Port 3000 (metrics endpoints)

## 🚨 Troubleshooting

### Common Issues

1. **Docker not installed**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

2. **Port conflicts**
   - Check if ports 3001 or 9090 are in use
   - Modify `docker-compose.monitoring.yml` to change ports

3. **Metrics not showing**
   - Ensure Flask app is running on port 3000
   - Check `/metrics` endpoint is accessible
   - Verify Prometheus can reach the Flask app

4. **Dashboard not loading**
   - Check Grafana logs: `docker-compose logs grafana`
   - Verify datasource connection in Grafana

### Useful Commands
```bash
# Check if containers are running
docker ps

# View Prometheus targets
curl http://localhost:9090/api/v1/targets

# Test metrics endpoint
curl http://localhost:3000/metrics

# Check Grafana health
curl http://localhost:3001/api/health
```

## 🔄 Migration from Flask-MonitoringDashboard

The migration to Grafana provides:
- ✅ Better visualization capabilities
- ✅ More flexible alerting
- ✅ Historical data retention
- ✅ Custom dashboard creation
- ✅ Integration with other monitoring tools
- ✅ Better performance and scalability

## 📚 Additional Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Prometheus Client Python](https://github.com/prometheus/client_python) 