@echo off

REM Go to project root
cd /d %~dp0

REM Create virtual environment if missing
IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run application
echo Starting webcam application...
python main.py

pause