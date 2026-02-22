@echo off
echo ============================================================
echo VoltNet - Train Model (Dummy Data for Testing)
echo ============================================================
echo.
echo NOTE: Currently using dummy data for testing
echo Real IEEE dataset loading will be implemented next
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo ============================================================
echo Training on Dummy Dataset (39 nodes)
echo ============================================================
echo   - 900 samples
echo   - 50 epochs (~10-15 minutes)
echo   - Tests the training pipeline
echo.
pause

python train_real_model.py --bus_system ieee39 --epochs 50

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
echo.
echo Results saved in:
echo   models/best_model.pth
echo   models/training_history.json
echo.
echo Artifacts deployed to:
echo   backend/artifacts/
echo.
echo Next: Start backend
echo   cd backend
echo   python app.py
echo.
pause
