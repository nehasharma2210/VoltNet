@echo off
echo 🚀 Starting VoltNet deployment...

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker-compose down

REM Build and start containers
echo 🔨 Building and starting containers...
docker-compose up --build -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak > nul

REM Check if services are running
echo 🔍 Checking service status...
docker-compose ps

REM Test backend health
echo 🏥 Testing backend health...
curl -f http://localhost:8000/health

REM Show logs if needed
echo 📋 Recent logs:
docker-compose logs --tail=20

echo ✅ Deployment complete!
echo 🌐 Frontend: http://localhost
echo 🔧 Backend API: http://localhost:8000
echo 📊 Health check: http://localhost:8000/health

pause