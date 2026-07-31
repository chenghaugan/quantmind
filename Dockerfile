FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# 系统依赖 + 中文时区（真实行情时间处理必须）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git tzdata \
    && ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 拷全部源码（开发时由 compose 卷挂载覆盖，实现热更新）
COPY . .

# 安装项目及全部依赖（editable，确保 import 始终指向 /app/quantmind）
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

EXPOSE 8000 8501

# 默认起 API；compose 可用 command 覆盖为 streamlit
CMD ["uvicorn", "quantmind.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
