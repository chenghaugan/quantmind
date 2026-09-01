"""SettingsService：AI 模型配置（API Key / Base URL / 模型 / 温度）的持久化与运行时生效。

- 默认值来自环境变量 ``QM_LLM_*``（见 ``config.Settings``）。
- 用户通过「设置」页保存后写入 ``config/ai_settings.json``，并在不重启服务的情况下
  重建 ``ResearchService`` 使用的 LLM Provider。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from ...config import get_settings
from ...ai import build_provider

# 允许前端更新的字段白名单
_ALLOWED = {"provider", "api_key", "base_url", "model", "temperature", "timeout"}

# 字段 -> 环境变量名（与 config.Settings 的 env_prefix="QM_" 对应）
_ENV_MAP = {
    "provider": "QM_LLM_PROVIDER",
    "api_key": "QM_LLM_API_KEY",
    "base_url": "QM_LLM_BASE_URL",
    "model": "QM_LLM_MODEL",
    "temperature": "QM_LLM_TEMPERATURE",
    "timeout": "QM_LLM_TIMEOUT",
}


class SettingsService:
    def __init__(self) -> None:
        # 配置目录：quantmind/config/ai_settings.json
        self.path = Path(__file__).resolve().parent.parent.parent / "config" / "ai_settings.json"
        self.data: Dict[str, Any] = self._load()

    # ----------------------------------------------------------------- 读取
    def _defaults(self) -> Dict[str, Any]:
        """配置默认值，优先级：进程环境变量 > 项目根 .env > 代码默认值。

        pydantic-settings 只在 cwd 恰好是项目根时才会读到 .env，这里显式解析
        .env 文件保证任何 cwd 下都能拿到相同回退逻辑。
        """
        s = get_settings()
        dot = self._read_dotenv()
        d: Dict[str, Any] = {}
        for field, env_name in _ENV_MAP.items():
            if os.environ.get(env_name):
                val: Any = os.environ[env_name]
            elif env_name in dot:
                val = dot[env_name]
            else:
                val = getattr(s, f"llm_{field}")
            if field == "temperature":
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 0.7
            elif field == "timeout":
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 120.0
            d[field] = val
        return d

    def _load(self) -> Dict[str, Any]:
        data = self._defaults()
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    data.update({k: v for k, v in saved.items() if k in _ALLOWED})
            except Exception:  # noqa: BLE001
                pass
        return data

    def get(self) -> Dict[str, Any]:
        out = dict(self.data)
        out["source"] = self.source()
        return out

    # ----------------------------------------------------------------- .env 同步
    def _env_path(self) -> Path:
        """与 config.Settings(env_file='.env') 对齐：相对进程 cwd 的项目根 .env。"""
        return Path.cwd() / ".env"

    def _read_dotenv(self) -> Dict[str, str]:
        """读取项目根 .env 中 QM_LLM_* 键值（不依赖 python-dotenv，手写解析）。"""
        p = self._env_path()
        vals: Dict[str, str] = {}
        if not p.exists():
            return vals
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in _ENV_MAP.values():
                vals[k] = v
        return vals

    def _sync_env(self) -> bool:
        """把当前 5 个 AI 配置键合并写入项目根 .env（保留其它键与注释）。

        返回是否成功写入。UI 保存时调用，使配置在重启 / 容器部署时仍通过
        环境变量生效，从而与 json 持久化形成双向同步。
        """
        try:
            p = self._env_path()
            new_vals = {_ENV_MAP[k]: str(self.data.get(k, "")) for k in _ENV_MAP}
            if p.exists():
                lines = p.read_text(encoding="utf-8").splitlines()
            else:
                lines = ["# QuantMind AI 模型配置（由「设置」页自动同步，请勿手改亦可）"]
            kept = [
                ln for ln in lines
                if not any(ln.strip().startswith(e + "=") for e in _ENV_MAP.values())
            ]
            body = list(kept)
            if body and body[-1].strip() != "":
                body.append("")
            for env_key, val in new_vals.items():
                # 值含特殊字符时加双引号，兼容大多数加载器
                safe = f'{env_key}="{val}"' if (val and (" " in val or "#" in val)) else f"{env_key}={val}"
                body.append(safe)
            while body and body[-1].strip() == "":
                body.pop()
            p.write_text("\n".join(body) + "\n", encoding="utf-8")
            return True
        except Exception:  # noqa: BLE001
            return False

    def source(self) -> str:
        """配置来源：json 文件 / 环境变量 .env / 代码默认值。"""
        if self.path.exists():
            return "json"
        if self._read_dotenv():
            return "env"
        return "default"

    # ----------------------------------------------------------------- 保存
    def save(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        for key in _ALLOWED:
            if key in payload:
                self.data[key] = payload[key]
        # 数值字段清洗
        try:
            self.data["temperature"] = float(self.data.get("temperature", 0.7))
        except (TypeError, ValueError):
            self.data["temperature"] = 0.7
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 双向同步：同时写入项目根 .env（保留其它键）
        synced = self._sync_env()
        out = dict(self.data)
        out["synced_env"] = synced
        return out

    # ----------------------------------------------------------------- 生效
    def rebuild_provider(self):
        return build_provider(
            self.data.get("provider", "mock"),
            self.data.get("api_key", ""),
            self.data.get("base_url", ""),
            self.data.get("model", ""),
            float(self.data.get("temperature", 0.7)),
            timeout=float(self.data.get("timeout", 120.0)),
        )

    async def test(self, idea: str = "用一句话返回一个期货动量因子的名称。") -> Dict[str, Any]:
        """用当前 Provider 发一条测试请求，返回片段与 Provider 名称。"""
        provider = self.rebuild_provider()
        try:
            text = await provider.chat("", idea)
            return {"provider": provider.name, "ok": True, "sample": (text or "")[:300]}
        except Exception as e:  # noqa: BLE001
            return {"provider": provider.name, "ok": False, "error": str(e)[:300]}
