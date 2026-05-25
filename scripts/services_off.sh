#!/bin/bash
# 🛑 STOP all DocQuery Fargate services (saves ~$5-6/day)
# Usage: ./scripts/services_off.sh

CLUSTER="docquery-prod-Cluster-FbHScEllM2lH"
REGION="ap-south-1"

echo "🛑 Stopping all DocQuery services..."
echo ""

SERVICES=(
  "docquery-prod-api-Service-M6RwvqeZ0DQ3"
  "docquery-prod-worker-Service-b4jA6MKVchxi"
  "docquery-prod-frontend-Service-lKDIQFQHebgK"
  "docquery-prod-redis-Service-IVK5MWmqv5SZ"
)

NAMES=("API" "Worker" "Frontend" "Redis")

for i in "${!SERVICES[@]}"; do
  aws ecs update-service \
    --cluster "$CLUSTER" \
    --service "${SERVICES[$i]}" \
    --desired-count 0 \
    --region "$REGION" \
    --no-cli-pager \
    --output text \
    --query 'service.desiredCount' > /dev/null 2>&1
  
  if [ $? -eq 0 ]; then
    echo "  ✅ ${NAMES[$i]} → stopped (0 containers)"
  else
    echo "  ❌ ${NAMES[$i]} → failed to stop"
  fi
done

echo ""
echo "🎉 All services stopped! Container cost is now $0/hour."
echo "⚠️  Note: ALB + NAT Gateway still cost ~\$1.60/day even when services are off."
echo "   To eliminate ALL costs, run: copilot app delete"
