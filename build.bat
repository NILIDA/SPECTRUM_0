@echo off
rem This is deconvolution app build script

set app_name=infrared
set venv_dir=venv

rem Populate virtualenv and install dependencies
if not exist %venv_dir%\ (
  echo [INFO] Virtualenv folder '%venv_dir%' not found
  python -m virtualenv %venv_dir%
  call %venv_dir%\Scripts\activate.bat
  python -m pip install -r requirements.txt
)

rem Activate VirtualEnv
echo [INFO] Activate virtualenv "%venv_dir%"
call %venv_dir%\Scripts\activate.bat

rem Install PyInstaller dependencies
echo [INFO] Install PyInstaller and dependencies
python -m pip install pyinstaller pyinstaller-hooks-contrib

rem Pack application to exe
echo [INFO] Build and pack exe file
pyinstaller --onefile --noconsole --noconfirm --collect-data scienceplots --add-data icons:icons --name %app_name% main.py

rem Next line optional
echo [INFO] Execution finished
rem pause
