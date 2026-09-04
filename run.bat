@echo off
chcp 65001 > nul
echo ===================================================
echo     TeachYOLO — Запуск YOLO26 Studio (Windows)     
echo ===================================================

:: 1. Check if setup was run
if not exist .venv (
    echo [!] Виртуальное окружение не найдено. Запуск установки setup.bat...
    call setup.bat
)

:: 2. Activate virtual environment
echo [+] Активация окружения (.venv)...
call .venv\Scripts\activate.bat

:: 3. Create workspace directory if missing
if not exist workspace mkdir workspace

:: 4. Open default browser
echo [+] Запуск веб-интерфейса в браузере...
start http://localhost:8000

:: 5. Start Uvicorn server
echo [+] Сервер TeachYOLO запущен на http://localhost:8000
echo [+] Для остановки закройте это окно или нажмите Ctrl+C
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
