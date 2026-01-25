#!/bin/bash
echo "🚀 Starting VoltNet Backend on Render..."
echo "Port: $PORT"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Files in directory:"
ls -la

# Start the application
exec python -m uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1