#!/bin/bash
# 📊 Check status of all DocQuery Fargate services
# Usage: ./scripts/services_status.sh

CLUSTER="docquery-prod-Cluster-FbHScEllM2lH"
REGION="ap-south-1"

echo "📊 DocQuery Service Status"
echo "=========================="
echo ""

SERVICES=(
  "docquery-prod-api-Service-M6RwvqeZ0DQ3"
  "docquery-prod-worker-Service-b4jA6MKVchxi"
  "docquery-prod-frontend-Service-lKDIQFQHebgK"
  "docquery-prod-redis-Service-IVK5MWmqv5SZ"
)
NAMES=("API" "Worker" "Frontend" "Redis")

for i in "${!SERVICES[@]}"; do
  STATUS=$(aws ecs describe-services \
    --cluster "$CLUSTER" \
    --services "${SERVICES[$i]}" \
    --region "$REGION" \
    --no-cli-pager \
    --output text \
    --query 'services[0].[desiredCount,runningCount]' 2>/dev/null)
  
  DESIRED=$(echo "$STATUS" | awk '{print $1}')
  RUNNING=$(echo "$STATUS" | awk '{print $2}')
  
  if [ "$RUNNING" = "0" ] && [ "$DESIRED" = "0" ]; then
    echo "  🔴 ${NAMES[$i]}: OFF (0/0 containers)"
  elif [ "$RUNNING" = "$DESIRED" ]; then
    echo "  🟢 ${NAMES[$i]}: RUNNING ($RUNNING/$DESIRED containers)"
  else
    echo "  🟡 ${NAMES[$i]}: STARTING ($RUNNING/$DESIRED containers)"
  fi
done

echo ""
echo "---"
echo "Commands:  ./scripts/services_on.sh  |  ./scripts/services_off.sh"
