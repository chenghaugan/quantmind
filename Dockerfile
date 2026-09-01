FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# 使用阿里云 Debian 镜像（无需代理）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null; \
    sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null; \
    true

# 系统依赖 + 中文时区（真实行情时间处理必须）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git tzdata \
    && ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 拷全部源码（开发时由 compose 卷挂载覆盖，实现热更新）
COPY . .

# 使用阿里云 PyPI 镜像安装依赖
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

EXPOSE 8000 8501

# 默认起 API；compose 可用 command 覆盖为 streamlit
CMD ["uvicorn", "quantmind.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
