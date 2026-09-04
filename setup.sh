#!/usr/bin/env bash
set -e

echo "==================================================="
echo "   TeachYOLO — Установка и настройка (Linux/macOS) "
echo "==================================================="

# 1. Check Python
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python не найден! Установите Python 3.10 или новее."
    exit 1
fi

PY_VER=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[+] Обнаружен Python: версия $PY_VER ($PYTHON_CMD)"

# 2. Create Virtual Environment
if [ -d ".venv" ]; then
    echo "[+] Виртуальное окружение .venv уже существует."
else
    echo "[+] Создание виртуального окружения .venv..."
    $PYTHON_CMD -m venv .venv
fi

# 3. Activate & Install Dependencies
echo "[+] Активация виртуального окружения..."
source .venv/bin/activate

echo "[+] Обновление pip и установка зависимостей из requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create Workspace
mkdir -p workspace
touch workspace/.gitkeep

echo ""
echo "==================================================="
echo "  [SUCCESS] Установка успешно завершена!          "
echo "  Для запуска TeachYOLO выполните: ./run.sh         "
echo "==================================================="
