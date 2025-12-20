@echo off
set ENV=%1
if "%ENV%"=="" set ENV=venv
python -m venv %ENV%
%ENV%\Scripts\python.exe -m pip install --upgrade pip
%ENV%\Scripts\python.exe -m pip install -r requirements.txt
echo To activate the venv (cmd.exe): %ENV%\Scripts\activate.bat
