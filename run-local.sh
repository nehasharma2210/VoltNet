#!/bin/bash

echo "🚀 Starting VoltNet Local Development..."

# Start backend in background
echo "📡 Starting Backend on http://localhost:8000"
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Start frontend in background
echo "🌐 Starting Frontend on http://localhost:3000"
cd frontend
python -m http.server 3000 &
FRONTEND_PID=$!
cd ..

echo "✅ Both services started!"
echo "🌐 Frontend: http://localhost:3000"
echo "📡 Backend: http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo "🏥 Health: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user to stop
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait