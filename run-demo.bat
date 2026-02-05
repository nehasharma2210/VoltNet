@echo off
echo 🚀 Starting VoltNet DEMO for Presentation...

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install minimal dependencies
echo 📦 Installing minimal dependencies...
pip install fastapi uvicorn[standard] numpy pydantic

REM Start demo backend
echo 📡 Starting Demo Backend on http://localhost:8000
start "VoltNet Demo Backend" cmd /k "call venv\Scripts\activate.bat && cd backend && python app-demo.py"

REM Wait for backend to start
echo ⏳ Waiting for backend to start...
timeout /t 3 /nobreak > nul

REM Start frontend
echo 🌐 Starting Frontend on http://localhost:3000
start "VoltNet Frontend" cmd /k "cd frontend && python -m http.server 3000"

echo ✅ Demo ready for presentation!
echo 🌐 Frontend: http://localhost:3000
echo 📡 Backend: http://localhost:8000
echo 📊 API Docs: http://localhost:8000/docs
echo 🏥 Health: http://localhost:8000/health
echo ""
echo 🎯 This is DEMO mode with simulated data - perfect for presentation!

pause