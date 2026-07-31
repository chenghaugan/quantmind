#!/usr/bin/env bash
# 在 quantmind 容器内跑真实数据。
#
# ⚠️ 前置：容器内默认网络无法直连 akshare/东财等数据源，必须经宿主机代理。
#   本机 clash 监听 7897；默认采用「直连 clash」方式（用户已确认「直连即可」）：
#     → 在 clash 设置里打开「允许局域网连接」(Allow LAN)，
#       容器即可经 host.docker.internal:7897 直连 clash，无需 Docker Desktop 代理。
#   备选：若不想开 clash LAN，也可走 Docker Desktop 代理——
#     Docker Desktop → Settings → Resources → Proxies 设 HTTP/HTTPS 代理为
#     http://127.0.0.1:7897，重启 Docker 后设 QM_DOCKER_PROXY=http://host.docker.internal:3128。
#
# 用法：
#   ./scripts/docker_realdata.sh python -m quantmind.cli backtest --symbol 600000 --exchange SSE --strategy dual_ma --cost
#   ./scripts/docker_realdata.sh python -m quantmind.cli backtest --symbol rb0 --exchange SHFE --strategy multifactor --cost
#   ./scripts/docker_realdata.sh python -m quantmind.cli cs --symbols rb0,hc0,bu0,i0 --exchange SHFE --name alpha021 --bt
set -euo pipefail

# 默认直连 clash（host.docker.internal:7897，需 clash 开 Allow LAN）；
# 如需改走 Docker Desktop 代理转发端口，设 QM_DOCKER_PROXY=http://host.docker.internal:3128
PROXY="${QM_DOCKER_PROXY:-http://host.docker.internal:7897}"

docker run --rm \
  -e "HTTP_PROXY=${PROXY}" -e "HTTPS_PROXY=${PROXY}" \
  -e "http_proxy=${PROXY}" -e "https_proxy=${PROXY}" \
  -e "QM_LLM_PROVIDER=${QM_LLM_PROVIDER:-mock}" \
  quantmind:latest "$@"
