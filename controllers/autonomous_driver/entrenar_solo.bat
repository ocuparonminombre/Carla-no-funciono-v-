@echo off
title Entrenando IA BMW X5
echo =================================================
echo   INICIANDO BUCLE DE ENTRENAMIENTO INFINITO
echo   Para detenerlo: Cierra esta ventana o presiona Ctrl+C
echo =================================================

:bucle
:: Ejecuta el driver. Si choca, el script de Python se cierra con sys.exit()
python controllers/autonomous_driver/main_driver.py

:: Espera 2 segundos antes de reabrirlo (para dar tiempo a Webots)
timeout /t 2 >nul

:: Vuelve a empezar
goto bucle