@echo off
cd /d "%~dp0"
call run_deep_learning.bat %*
exit /b %errorlevel%
