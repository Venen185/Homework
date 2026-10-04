#!/usr/bin/env bash
# Запуск детектора паттернов Нисона (macOS / Linux). Без параметров = 15 голубых фишек.
# Пример: ./run.sh SBER --stats
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
  echo "Python не найден. Установите его с https://www.python.org/downloads/"
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "Первый запуск: создаю окружение и ставлю библиотеки, это займёт минуту..."
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt
.venv/bin/python nison_detector.py "$@"
