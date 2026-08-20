@echo off
REM ============================================================================
REM run_quantmind.bat - 一键启动 QuantMind 本地开发栈（后端 API + Web 前端）
REM
REM 必须从本脚本所在目录（即项目根 E:\...\量化投资经理\quantmind）启动，
REM 否则 KnowledgeStore() 默认 DB 会按 cwd 解析到错误位置，连到空的
REM knowledge.db，因子/策略在 Web 上看不到。
REM 本脚本用 %~dp0 绝对定位到脚本所在目录，不依赖调用者的 cwd 或 cd。
REM
REM 启动内容：
REM   1. 后端 API : %~dp0.venv\Scripts\python.exe -m uvicorn quantmind.api.app:app --host 127.0.0.1 --port 8000
REM   2. Web 前端 : %~dp0.venv\Scripts\python.exe -m streamlit run quantmind\web\streamlit_app.py
REM
REM 若 8000 端口已被监听（已有后端在跑），则跳过重复启动。
REM 本脚本只负责拉起两个进程并在控制台打印地址，不做复杂进程管理/停止逻辑。
REM ============================================================================

setlocal
REM 绝对定位到脚本所在目录 = 项目根（%~dp0 自带结尾反斜杠）
cd /d "%~dp0"
set "PROJECT_ROOT=%~dp0"
set "PY=%PROJECT_ROOT%.venv\Scripts\python.exe"
set "STREAMLIT=%PROJECT_ROOT%quantmind\web\streamlit_app.py"

if not exist "%PY%" (
    echo [ERROR] 未找到虚拟环境 Python: %PY%
    echo 请先运行 scripts\bootstrap_windows.bat 安装依赖。
    exit /b 1
)

if not exist "%STREAMLIT%" (
    echo [ERROR] 未找到 Streamlit 入口: %STREAMLIT%
    exit /b 1
)

echo 项目根: %PROJECT_ROOT%

REM ---- 1) 后端 API ----
netstat -an | findstr /R /C:":8000 .*LISTENING" >nul
if %errorlevel%==0 (
    echo [SKIP] 8000 端口已在监听（后端可能已在运行），跳过后端启动。http://127.0.0.1:8000
) else (
    echo [START] 后端 API 启动中: uvicorn quantmind.api.app:app @ http://127.0.0.1:8000
    start "" /b "%PY%" -m uvicorn quantmind.api.app:app --host 127.0.0.1 --port 8000
    timeout /t 2 /nobreak >nul
)

REM ---- 2) Web 前端 ----
echo [START] Web 前端启动中: streamlit run "%STREAMLIT%" @ http://127.0.0.1:8501
start "" /b "%PY%" -m streamlit run "%STREAMLIT%"
timeout /t 2 /nobreak >nul

REM ---- 3) 打印访问地址 ----
echo.
echo ============================================================
echo   QuantMind 已启动
echo   API  Swagger  : http://127.0.0.1:8000/docs
echo   Web  前端     : http://127.0.0.1:8501
echo ============================================================
echo.
echo 提示：若 Web 上因子/策略看不见，请确认本次是从项目根 %~dp0 启动。
echo （KnowledgeStore 默认库解析到 %PROJECT_ROOT%db\knowledge.db，从其他 cwd 启动会连到空库。）
echo.

endlocal
