#!/bin/bash

echo "🚀 Starting VoltNet deployment..."

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Remove old images (optional - uncomment if you want fresh builds)
# echo "🗑️ Removing old images..."
# docker-compose down --rmi all

# Build and start containers
echo "🔨 Building and starting containers..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
echo "🔍 Checking service status..."
docker-compose ps

# Test backend health
echo "🏥 Testing backend health..."
curl -f http://localhost:8000/health || echo "❌ Backend health check failed"

# Show logs if needed
echo "📋 Recent logs:"
docker-compose logs --tail=20

echo "✅ Deployment complete!"
echo "🌐 Frontend: http://localhost"
echo "🔧 Backend API: http://localhost:8000"
echo "📊 Health check: http://localhost:8000/health"