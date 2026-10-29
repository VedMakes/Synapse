@echo off
setlocal
cd /d "%~dp0\.."
echo ======================================================
echo   SYNAPSE GP LIVE ONNX HEAD-TO-HEAD BROADCAST
echo ======================================================
python renderer\run_live_broadcast.py --open-browser %*

