@echo off
echo 🔧 Setting up VoltNet Local Development...

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install backend dependencies
echo 📦 Installing backend dependencies...
pip install -r backend\requirements.txt

echo ✅ Setup complete!
echo 🚀 Now run: run-local.bat

pause