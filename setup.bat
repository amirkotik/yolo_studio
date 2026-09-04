@echo off
chcp 65001 > nul
echo ===================================================
echo     TeachYOLO — Установка и настройка (Windows)    
echo ===================================================

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не найден! Установите Python 3.10 или новее с официального сайта python.org и добавьте его в PATH.
    pause
    exit /b 1
)

echo [+] Обнаружен Python.

:: 2. Create Virtual Environment
if exist .venv (
    echo [+] Виртуальное окружение .venv уже существует.
) else (
    echo [+] Создание виртуального окружения .venv...
    python -m venv .venv
)

:: 3. Activate & Install Dependencies
echo [+] Активация виртуального окружения...
call .venv\Scripts\activate.bat

echo [+] Обновление pip и установка зависимостей из requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. Create Workspace
if not exist workspace mkdir workspace

echo.
echo ===================================================
echo   [SUCCESS] Установка успешно завершена!          
echo   Для запуска TeachYOLO выполните файл: run.bat    
echo ===================================================
pause
