@echo off
setlocal
cd /d "%~dp0"

set VENV_NAME=.venv
set VENV_PATH=%VENV_NAME%\Scripts\activate.bat

REM --- Revisar o Crear el entorno virtual ---
IF EXIST "%VENV_PATH%" (
    echo El entorno virtual "%VENV_NAME%" ya existe.
) ELSE (
    echo El entorno virtual "%VENV_NAME%" no existe. Creando...
    python -m venv %VENV_NAME%
    IF ERRORLEVEL 1 (
        echo ERROR: No se pudo crear el entorno virtual.
        goto :error_end
    )
    echo Entorno virtual creado con exito.
)

REM --- Activar e Instalar dependencias ---
call "%VENV_PATH%"
IF EXIST "requirements.txt" (
    echo Instalando dependencias...
    pip install -r requirements.txt
    IF ERRORLEVEL 1 (
        echo ERROR: Fallo la instalacion de dependencias.
        goto :error_end
    )
    echo Dependencias instaladas/verificadas.
) ELSE (
    echo Advertencia: No se encontro requirements.txt.
)

:success_end
echo.
echo Setup finalizado correctamente.
pause
exit /b 0

:error_end
echo.
echo Hubo un error en el proceso de instalacion.
pause
exit /b 1