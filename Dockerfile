FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 编译依赖（akshare/mootdx/pydantic 等纯 Python，但部分需 build-essential）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# 用 uv 加速依赖安装（带依赖层缓存）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 先拷依赖清单，利用层缓存；仅当依赖变更时才重装
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen 2>/dev/null || uv pip install --system . || pip install --no-cache-dir .

# 再拷源码（开发时由 compose 卷挂载覆盖，此处用于镜像内完整副本）
COPY . .

EXPOSE 8000 8501

# 默认起 API；compose 可用 command 覆盖为 streamlit
CMD ["uvicorn", "quantmind.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
