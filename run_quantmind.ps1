# ============================================================================
# run_quantmind.ps1 — 一键启动 QuantMind 本地开发栈（后端 API + Web 前端）
#
# 必须从本脚本所在目录（即项目根 E:\...\量化投资经理\quantmind）启动，
# 否则 KnowledgeStore() 默认 DB 会按 __file__ 之外的 cwd 解析到错误位置，
# 导致连到一个空的 knowledge.db，因子/策略在 Web 上看不到。
# 本脚本用 $PSScriptRoot 绝对定位到脚本所在目录，不依赖调用者的 cwd 或 cd。
#
# 启动内容：
#   1. 后端 API   : <venv>\python.exe -m uvicorn quantmind.api.app:app --host 127.0.0.1 --port 8000
#   2. Web 前端   : <venv>\python.exe -m streamlit run quantmind\web\streamlit_app.py
#
# 后端独立后台运行；若 8000 端口已被监听（已有后端在跑），则跳过重复启动。
# 本脚本只负责拉起两个进程并在控制台打印访问地址，不做复杂进程管理/停止逻辑。
# ============================================================================

# 绝对定位到脚本所在目录 = 项目根
Set-Location -Path $PSScriptRoot
$ProjectRoot = $PSScriptRoot

$Py = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Streamlit = Join-Path $ProjectRoot 'quantmind\web\streamlit_app.py'

if (-not (Test-Path -LiteralPath $Py)) {
    Write-Host "[ERROR] 未找到虚拟环境 Python: $Py" -ForegroundColor Red
    Write-Host "请先运行 scripts\bootstrap_windows.bat 安装依赖。"
    exit 1
}

if (-not (Test-Path -LiteralPath $Streamlit)) {
    Write-Host "[ERROR] 未找到 Streamlit 入口: $Streamlit" -ForegroundColor Red
    exit 1
}

Write-Host "项目根: $ProjectRoot" -ForegroundColor Cyan

# ---- 1) 后端 API ----
$ApiBase = 'http://127.0.0.1:8000'
$PortListening = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($PortListening) {
    Write-Host "[SKIP] 8000 端口已在监听（后端可能已在运行），跳过后端启动。$ApiBase" -ForegroundColor Yellow
} else {
    Write-Host "[START] 后端 API 启动中: uvicorn quantmind.api.app:app @ $ApiBase" -ForegroundColor Green
    Start-Process -FilePath $Py -ArgumentList @(
        '-m','uvicorn','quantmind.api.app:app','--host','127.0.0.1','--port','8000'
    ) -WorkingDirectory $ProjectRoot -WindowStyle Hidden | Out-Null
    Start-Sleep -Seconds 2
}

# ---- 2) Web 前端 ----
$WebBase = 'http://127.0.0.1:8501'
Write-Host "[START] Web 前端启动中: streamlit run $Streamlit @ $WebBase" -ForegroundColor Green
Start-Process -FilePath $Py -ArgumentList @(
    '-m','streamlit','run', $Streamlit
) -WorkingDirectory $ProjectRoot -WindowStyle Hidden | Out-Null
Start-Sleep -Seconds 2

# ---- 3) 打印访问地址 ----
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  QuantMind 已启动" -ForegroundColor Green
Write-Host "  API  Swagger  : $ApiBase/docs" -ForegroundColor Cyan
Write-Host "  Web 前端      : $WebBase" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "提示：若 Web 上因子/策略看不见，请确认本次是从项目根 $PSScriptRoot 启动。" -ForegroundColor Yellow
Write-Host "（KnowledgeStore 默认库解析到 $ProjectRoot\db\knowledge.db，从其他 cwd 启动会连到空库。）"
Write-Host ""
