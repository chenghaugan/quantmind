#!/usr/bin/env bash
# 在 quantmind 容器内跑真实数据。
#
# ⚠️ 前置：容器内默认网络无法直连 akshare/东财等数据源，必须经宿主机代理。
#   本机 clash 监听 7897（该端口已被 clash 占用，容器侧不要再占用它）。
#   二选一打通（推荐方案 B，容器不直接依赖 clash 的具体端口）：
#   A) 直连 clash：clash 设置里打开「允许局域网连接」(Allow LAN)，然后运行本脚本时
#      设 QM_DOCKER_PROXY=http://host.docker.internal:7897 覆盖默认代理；
#   B) 经 Docker Desktop 代理（推荐）：Docker Desktop → Settings → Resources →
#      Proxies 把 HTTP/HTTPS 代理设为 http://127.0.0.1:7897（指向 clash），重启
#      Docker 后本脚本默认代理 http://host.docker.internal:3128 即可用；构建镜像
#      时 Docker daemon 也会走该代理联网，pip 安装依赖不再卡住。
#
# 用法：
#   ./scripts/docker_realdata.sh python -m quantmind.cli backtest --symbol 600000 --exchange SSE --strategy dual_ma --cost
#   ./scripts/docker_realdata.sh python -m quantmind.cli backtest --symbol rb0 --exchange SHFE --strategy multifactor --cost
#   ./scripts/docker_realdata.sh python -m quantmind.cli cs --symbols rb0,hc0,bu0,i0 --exchange SHFE --name alpha021 --bt
set -euo pipefail

# 默认走 Docker Desktop 代理转发端口 3128（避开 clash 占用的 7897）；
# 若 clash 已开「允许局域网连接」，可用 QM_DOCKER_PROXY 直连 7897：
#   QM_DOCKER_PROXY=http://host.docker.internal:7897 ./scripts/docker_realdata.sh ...
PROXY="${QM_DOCKER_PROXY:-http://host.docker.internal:3128}"

docker run --rm \
  -e "HTTP_PROXY=${PROXY}" -e "HTTPS_PROXY=${PROXY}" \
  -e "http_proxy=${PROXY}" -e "https_proxy=${PROXY}" \
  -e "QM_LLM_PROVIDER=${QM_LLM_PROVIDER:-mock}" \
  quantmind:latest "$@"
