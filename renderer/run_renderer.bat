@echo off
setlocal
cd /d "%~dp0"
echo ======================================================
echo   SYNAPSE GP 2.5D BROADCAST RENDERER
echo   Serving from: %CD%
echo   Live UI: http://127.0.0.1:8765/
echo ======================================================
start http://127.0.0.1:8765/
python -m http.server 8765 --bind 127.0.0.1
