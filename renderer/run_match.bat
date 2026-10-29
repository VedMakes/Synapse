@echo off
setlocal
cd /d "%~dp0\.."
echo ======================================================
echo   SYNAPSE GP KNOCKOUT MATCH EVALUATOR
echo ======================================================
python renderer\run_match.py --open-browser %*
