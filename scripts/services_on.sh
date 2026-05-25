#!/bin/bash
# 🟢 START all DocQuery Fargate services
# Usage: ./scripts/services_on.sh
# Services typically take 60-90 seconds to become healthy after starting.

CLUSTER="docquery-prod-Cluster-FbHScEllM2lH"
REGION="ap-south-1"

echo "🟢 Starting all DocQuery services..."
echo ""

# Start Redis first (other services depend on it)
echo "  ⏳ Starting Redis..."
aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "docquery-prod-redis-Service-IVK5MWmqv5SZ" \
  --desired-count 1 \
  --region "$REGION" \
  --no-cli-pager \
  --output text \
  --query 'service.desiredCount' > /dev/null 2>&1
echo "  ✅ Redis → starting (1 container)"

# Wait for Redis to be ready before starting dependent services
echo "  ⏳ Waiting 30s for Redis to be ready..."
sleep 30

# Start API, Worker, Frontend
SERVICES=(
  "docquery-prod-api-Service-M6RwvqeZ0DQ3"
  "docquery-prod-worker-Service-b4jA6MKVchxi"
  "docquery-prod-frontend-Service-lKDIQFQHebgK"
)
NAMES=("API" "Worker" "Frontend")

for i in "${!SERVICES[@]}"; do
  aws ecs update-service \
    --cluster "$CLUSTER" \
    --service "${SERVICES[$i]}" \
    --desired-count 1 \
    --region "$REGION" \
    --no-cli-pager \
    --output text \
    --query 'service.desiredCount' > /dev/null 2>&1
  
  if [ $? -eq 0 ]; then
    echo "  ✅ ${NAMES[$i]} → starting (1 container)"
  else
    echo "  ❌ ${NAMES[$i]} → failed to start"
  fi
done

echo ""
echo "🟢 All services starting! They'll be healthy in ~60-90 seconds."
echo ""
echo "📊 To check status:"
echo "   copilot svc status --name api"
echo "   copilot svc status --name worker"
