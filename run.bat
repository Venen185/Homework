@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem Запуск детектора паттернов Нисона. Двойной щелчок = 15 голубых фишек.
rem Параметры можно передать из командной строки: run.bat SBER --stats

set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto nopython

if not exist ".venv\Scripts\python.exe" (
    echo Первый запуск: создаю окружение и ставлю библиотеки, это займёт минуту...
    %PY% -m venv .venv || goto failed
)
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt || goto failed

".venv\Scripts\python.exe" nison_detector.py %*
echo.
pause
exit /b

:nopython
echo Python не найден. Установите его с https://www.python.org/downloads/
echo и при установке отметьте галочку "Add python.exe to PATH".
pause
exit /b 1

:failed
echo Не удалось установить библиотеки. Проверьте интернет и попробуйте ещё раз.
pause
exit /b 1
