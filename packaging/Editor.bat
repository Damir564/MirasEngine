@echo off
rem Starts MirasEngine in editor mode. Extra arguments are passed through.
start "" "%~dp0engine.exe" --editor %*
