@echo off
cd /d "%~dp0\Synapse\renderer" 2>nul || cd /d "%~dp0\renderer"
echo Starting SYNAPSE GP 2.5D Renderer on http://127.0.0.1:8765/ ...
start http://127.0.0.1:8765/
python -m http.server 8765

