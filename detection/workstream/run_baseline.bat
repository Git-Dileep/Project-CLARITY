@echo off
cd /d "%~dp0"
call run_traditional.bat %*
exit /b %errorlevel%
