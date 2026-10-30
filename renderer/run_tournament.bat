@echo off
setlocal enabledelayedexpansion

title Synapse GP Tournament Manager

cd /d "%~dp0\.."

echo ====================================================================
echo   SYNAPSE GRAND PRIX TOURNAMENT WORKFLOW MANAGER
echo ====================================================================
echo   [1] Generate Full Replays for All Teams in artifacts/onnx-packages
echo   [2] Stage Knockout Match (Team Alpha vs Team Beta)
echo   [3] View Current Monaco Leaderboard
echo   [4] Launch 3D Tournament Web Broadcast
echo   [5] Exit
echo ====================================================================

set /p CHOICE="Select an option (1-5): "

if "%CHOICE%"=="1" (
    echo.
    echo Simulating all entrant packages until full lap completion...
    python renderer/tournament.py simulate --all artifacts/onnx-packages --track monaco
    pause
    goto :eof
)

if "%CHOICE%"=="2" (
    echo.
    set /p TEAM_A="Enter Team 1 name (e.g. team_alpha): "
    set /p TEAM_B="Enter Team 2 name (e.g. team_beta): "
    python renderer/tournament.py match !TEAM_A! !TEAM_B! --track monaco --open-browser
    pause
    goto :eof
)

if "%CHOICE%"=="3" (
    echo.
    python renderer/tournament.py leaderboard --track monaco
    pause
    goto :eof
)

if "%CHOICE%"=="4" (
    echo.
    echo Opening 3D Replay Broadcast...
    start http://127.0.0.1:8765/?match=true
    goto :eof
)

if "%CHOICE%"=="5" (
    goto :eof
)

echo Invalid choice.
pause

