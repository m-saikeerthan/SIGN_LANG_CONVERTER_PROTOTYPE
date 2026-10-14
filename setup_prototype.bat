@echo off
REM ==============================================================================
REM ISL Prototype Setup Script for Windows
REM ==============================================================================

echo ======================================================================
echo   Indian Sign Language (ISL) Prototype Windows Setup
echo ======================================================================

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10 - 3.12 from python.org and check "Add to PATH".
    pause
    exit /b 1
)

echo [1/3] Upgrading pip and installing requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo [2/3] Checking .env configuration...
if not exist ".env" (
    echo Creating default .env template...
    echo GEMINI_API_KEY=> .env
    echo [NOTE] A default .env was created. You can optionally add your GEMINI_API_KEY for cloud NLP.
)

echo [3/3] Running Prototype Diagnostic Suite...
python diagnose_prototype.py --no-audio

echo ======================================================================
echo   Setup Finished!
echo.
echo   To run the Web Application:
echo       python web\app.py
echo.
echo   To run Desktop Live Detection:
echo       python src\isl_predictor.py
echo.
echo   To run the Headless Prototype Engine:
echo       python src\headless_prototype.py
echo ======================================================================
pause
