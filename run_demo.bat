@echo off
setlocal
cd /d "%~dp0"

set VENV_NAME=.venv
set VENV_PATH=%VENV_NAME%\Scripts\activate.bat

REM --- Validar que el ambiente exista ---
IF NOT EXIST "%VENV_PATH%" (
    echo ERROR: El entorno virtual no existe. 
    echo Por favor, ejecuta primero 'setup_venv.bat'.
    goto :error_end
)

REM --- Activar y Ejecutar ---
call "%VENV_PATH%"
echo Iniciando aplicacion...
cls

python demo.py
IF ERRORLEVEL 1 (
    echo.
    echo ERROR: La aplicacion se cerro con un codigo de error.
    goto :error_end
)

goto :eof

:error_end
echo.
echo No se pudo iniciar el script.
pause
exit /b 1