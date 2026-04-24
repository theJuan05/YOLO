@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting YOLO Flask app...
python app.py
pause
