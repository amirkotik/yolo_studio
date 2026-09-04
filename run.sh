#!/usr/bin/env bash
set -e

echo "==================================================="
echo "   TeachYOLO — Запуск YOLO26 Studio (Linux/macOS)  "
echo "==================================================="

# 1. Check if setup was run
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo "[!] Виртуальное окружение не найдено. Автоматический запуск ./setup.sh..."
    chmod +x setup.sh
    ./setup.sh
fi

# 2. Activate virtual environment
if [ -d ".venv" ]; then
    VENV_DIR=".venv"
else
    VENV_DIR="venv"
fi

echo "[+] Активация окружения ($VENV_DIR)..."
source "$VENV_DIR/bin/activate"

# 3. Create workspace directory if missing
mkdir -p workspace

# 4. Open default browser after a short delay
echo "[+] Запуск веб-интерфейса в браузере..."
(
    sleep 2
    if command -v xdg-open > /dev/null; then
        xdg-open http://localhost:8000 2>/dev/null || true
    elif command -v open > /dev/null; then
        open http://localhost:8000 2>/dev/null || true
    fi
) &

# 5. Start Uvicorn server
echo "[+] Сервер TeachYOLO запущен на http://localhost:8000"
echo "[+] Для остановки нажмите Ctrl+C"
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
