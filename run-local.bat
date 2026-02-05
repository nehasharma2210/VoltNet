@echo off
echo 🚀 Starting VoltNet Local Development...

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if needed
echo 📦 Installing dependencies...
pip install -r backend\requirements.txt

REM Start backend in new window
echo 📡 Starting Backend on http://localhost:8000
start "VoltNet Backend" cmd /k "call venv\Scripts\activate.bat && cd backend && python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for backend to start
echo ⏳ Waiting for backend to start...
timeout /t 5 /nobreak > nul

REM Start frontend in new window
echo 🌐 Starting Frontend on http://localhost:3000
start "VoltNet Frontend" cmd /k "cd frontend && python -m http.server 3000"

echo ✅ Both services started!
echo 🌐 Frontend: http://localhost:3000
echo 📡 Backend: http://localhost:8000
echo 📊 API Docs: http://localhost:8000/docs
echo 🏥 Health: http://localhost:8000/health

pause