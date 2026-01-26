#!/bin/bash
set -e

echo "🚀 Starting VoltNet on Render..."
echo "Port: $PORT"
echo "Working directory: $(pwd)"

# Update frontend config with backend URL
if [ -f "/app/frontend/config.js" ]; then
    echo "📝 Updating frontend config..."
    sed -i "s|http://backend:8000|http://localhost:$PORT|g" /app/frontend/config.js
    sed -i "s|https://voltnet-backend.onrender.com|http://localhost:$PORT|g" /app/frontend/config.js
fi

# Start nginx for frontend (in background)
echo "🌐 Starting nginx for frontend..."
nginx -g "daemon off;" &

# Wait a moment for nginx to start
sleep 2

# Start backend API
echo "🔧 Starting FastAPI backend..."
cd /app/backend
exec python -m uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1