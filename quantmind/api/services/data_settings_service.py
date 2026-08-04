"""DataSettingsService：本地数据路径配置的持久化与读取。

管理 5 个本地数据根目录：
  local_data_root / local_stock_root / local_hk_root / local_option_root / seat_data_root

读取优先链：运行时 JSON 覆盖文件(config/data_settings.json) > 环境变量 > 代码默认值。
保存时写入 JSON，并同步写回项目根 .env 的 ``QM_LOCAL_*`` 变量，使配置在重启后仍生效
（与 SettingsService 的 AI 配置双写策略一致）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ...config import get_settings

# 允许前端更新的本地数据路径字段白名单（与 config.Settings 字段一一对应）
_ALLOWED = [
    "local_data_root",
    "local_stock_root",
    "local_hk_root",
    "local_option_root",
    "seat_data_root",
]

# 字段 -> env 变量名（对应 config.Settings 的 QM_ 前缀 + pydantic 字段名自动映射）
_ENV_MAP = {
    "local_data_root": "QM_LOCAL_DATA_ROOT",
    "local_stock_root": "QM_LOCAL_STOCK_ROOT",
    "local_hk_root": "QM_LOCAL_HK_ROOT",
    "local_option_root": "QM_LOCAL_OPTION_ROOT",
    "seat_data_root": "QM_SEAT_DATA_ROOT",
}


def _read_dotenv() -> Dict[str, str]:
    """读取项目根 .env 中的 QM_LOCAL_* 键值（手写解析，不依赖 python-dotenv）。"""
    p = Path.cwd() / ".env"
    vals: Dict[str, str] = {}
    if not p.exists():
        return vals
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.startswith("QM_LOCAL_"):
            vals[k] = v
    return vals


class DataSettingsService:
    def __init__(self) -> None:
        self.path = Path(__file__).resolve().parent.parent.parent / "config" / "data_settings.json"
        self.data: Dict[str, str] = self._load()

    # ----------------------------------------------------------------- 读取
    def _defaults(self) -> Dict[str, str]:
        s = get_settings()
        dot = _read_dotenv()
        out: Dict[str, str] = {}
        for field, env_name in _ENV_MAP.items():
            if os.environ.get(env_name):
                out[field] = os.environ[env_name]
            elif env_name in dot:
                out[field] = dot[env_name]
            else:
                out[field] = getattr(s, field, "")
        return out

    def _load(self) -> Dict[str, str]:
        data = self._defaults()
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    data.update({k: str(v) for k, v in saved.items() if k in _ALLOWED})
            except Exception:  # noqa: BLE001
                pass
        return data

    def get(self) -> Dict[str, Any]:
        out = {k: self.data.get(k, "") for k in _ALLOWED}
        out["source"] = "json" if self.path.exists() else "env"
        return out

    # ----------------------------------------------------------------- 保存
    def _sync_env(self) -> bool:
        """把 5 个本地数据路径写回项目根 .env（保留其它键与注释）。"""
        try:
            p = Path.cwd() / ".env"
            new_vals = {_ENV_MAP[k]: str(self.data.get(k, "")) for k in _ALLOWED}
            if p.exists():
                lines = p.read_text(encoding="utf-8").splitlines()
            else:
                lines = ["# QuantMind 本地数据路径配置（由「设置」页自动同步，无需手改）"]
            kept = [
                ln for ln in lines
                if not any(ln.strip().startswith(e + "=") for e in _ENV_MAP.values())
            ]
            body = list(kept)
            if body and body[-1].strip() != "":
                body.append("")
            for env_key, val in new_vals.items():
                safe = f'{env_key}="{val}"' if (val and (" " in val or "#" in val)) else f"{env_key}={val}"
                body.append(safe)
            while body and body[-1].strip() == "":
                body.pop()
            p.write_text("\n".join(body) + "\n", encoding="utf-8")
            return True
        except Exception:  # noqa: BLE001
            return False

    def save(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        for key in _ALLOWED:
            if key in payload and payload[key] is not None:
                self.data[key] = str(payload[key])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        synced = self._sync_env()
        out = self.get()
        out["synced_env"] = synced
        return out
