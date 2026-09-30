@echo off
rem Starts MirasEngine in game mode. Drop a .scn file on this script to play it as the level.
if "%~1"=="" (
    start "" "%~dp0engine.exe" --game
) else (
    start "" "%~dp0engine.exe" --game --scene "%~f1"
)
