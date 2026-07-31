#!/usr/bin/env bash
# 在 quantmind 容器内跑真实数据。
#
# ⚠️ 前置：容器内默认网络无法直连 akshare/东财等数据源，必须经宿主机代理。
#   本机 clash 默认只监听 127.0.0.1，容器走不到。二选一打通：
#   A) clash 设置里打开「允许局域网连接」(Allow LAN)，然后本脚本默认代理
#      http://host.docker.internal:7897 即可用；
#   B) 在 Docker Desktop → Settings → Resources → Proxies 把
#      HTTP/HTTPS 代理设为 http://127.0.0.1:7897，重启 Docker 后改用
#      http://host.docker.internal:3128。
#
# 用法：
#   ./scripts/docker_realdata.sh python -m quantmind.cli backtest --symbol 600000 --exchange SSE --strategy dual_ma --cost
#   ./scripts/docker_realdata.sh python -m quantmind.cli backtest --symbol rb0 --exchange SHFE --strategy multifactor --cost
#   ./scripts/docker_realdata.sh python -m quantmind.cli cs --symbols rb0,hc0,bu0,i0 --exchange SHFE --name alpha021 --bt
set -euo pipefail

PROXY="${QM_DOCKER_PROXY:-http://host.docker.internal:7897}"

docker run --rm \
  -e "HTTP_PROXY=${PROXY}" -e "HTTPS_PROXY=${PROXY}" \
  -e "http_proxy=${PROXY}" -e "https_proxy=${PROXY}" \
  -e "QM_LLM_PROVIDER=${QM_LLM_PROVIDER:-mock}" \
  quantmind:latest "$@"
