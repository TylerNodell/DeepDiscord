#!/bin/bash

echo "🎯 Starting DeepDiscord GPU & Training Monitoring Dashboard..."

# Start Prometheus for metrics collection
echo "📊 Starting Prometheus..."
docker run -d --name deepdiscord-prometheus \
  -p 9090:9090 \
  --network bridge \
  -v /home/nuko/dev/DeepDiscord/training_module/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  --restart unless-stopped \
  prom/prometheus:latest

# Wali 2 seconds for Prometheus to initialize  
sleep 2

# Start GPU monitoring exporter
echo "🔥 Starting GPU Exporter..."
docker run -d --name deepdiscord-nvidia-exporter \
  -p 9445:9445 \
  --gpus all \
  --restart unless-stopped \
  mindprince/nvidia_gpu_prometheus_exporter:latest

# Start node exporter for system metrics
echo "💻 Starting System Metrics Exporter..."
docker run -d --name deepdiscord-node-exporter \
  -p 9100:9100 \
  --net="host" \
  --pid="host" \
  -v "/:/host:ro,rslave" \
  --restart unless-stopped \
  prom/node-exporter:latest \
  --path.rootfs=/host

# Wait for exporters to start
sleep 3

# Start Grafana dashboard
echo "📈 Starting Grafana Dashboard..."
docker run -d --name deepdiscord-grafana \
  -p 3000:3000 \
  --network bridge \
  -v grafana_data:/var/lib/grafana \
  -e GF_SECURITY_ADMIN_PASSWORD=deepdiscord \
  -e GF_USERS_ALLOW_SIGN_UP=false \
  --restart unless-stopped \
  grafana/grafana:latest

sleep 5

# Show access information
echo ""
echo "✅ Monitoring Stack Started Successfully!"
echo "==========================================>"
echo ""
echo "📊 Prometheus (Metrics): http://localhost:9090"
echo "📈 Grafana (Dashboards): http://localhost:3000"
echo "    - Username: admin" 
echo "    - Password: deepdiscord"
echo ""
echo "🔥 GPU Metrics: http://localhost:9445/metrics"
echo "💻 System Metrics: http://localhost:9100/metrics"
echo ""
echo "⚡ Add Prometheus data source in Grafana:"
echo "    URL: http://localhost:9090"
echo ""
echo "🎯 Your training container can be monitored through these dashboards!"
echo "   Container name: sherif-retrain"
echo ""