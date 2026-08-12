"""ProfileService：投资者偏好画像的 CRUD 与持久化。

- 预定义 5 种 Profile（aggressive/balanced/conservative/growth/value）。
- 用户可创建自定义 Profile，保存在 ``config/profiles.json``。
- 预定义 Profile 不可删除/覆盖（仅可复制后修改为自定义）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.profile import InvestorProfile, PRESET_PROFILES

_logger = logging.getLogger("quantmind.api.profile_service")


class ProfileService:
    """投资者偏好画像服务。"""

    def __init__(self) -> None:
        self.path = Path(__file__).resolve().parent.parent.parent / "config" / "profiles.json"
        self._custom: Dict[str, InvestorProfile] = {}
        self._load()

    # ----------------------------------------------------------------- 加载
    def _load(self) -> None:
        """从 JSON 文件加载自定义 Profile。"""
        self._custom = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for p in data.get("custom", []):
                    profile = InvestorProfile.from_dict(p)
                    # 不允许覆盖预定义 ID
                    if profile.id not in PRESET_PROFILES:
                        self._custom[profile.id] = profile
            except Exception as exc:
                _logger.warning("加载 profiles.json 失败: %s", exc)

    def _save(self) -> None:
        """持久化自定义 Profile 到 JSON 文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "custom": [p.to_dict() for p in self._custom.values()],
        }
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ----------------------------------------------------------------- CRUD
    def list_profiles(self) -> List[Dict[str, Any]]:
        """列出所有 Profile（预定义 + 自定义）。"""
        all_profiles = {**PRESET_PROFILES, **self._custom}
        return [p.to_dict() for p in all_profiles.values()]

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """读取单个 Profile。"""
        all_profiles = {**PRESET_PROFILES, **self._custom}
        p = all_profiles.get(profile_id)
        return p.to_dict() if p else None

    def create_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """创建自定义 Profile。

        不允许与预定义 ID 重名。
        """
        pid = payload.get("id", "").strip()
        if not pid:
            raise ValueError("Profile id 不能为空")
        if pid in PRESET_PROFILES:
            raise ValueError(f"不能与预定义 Profile 重名: {pid}")
        if pid in self._custom:
            raise ValueError(f"自定义 Profile 已存在: {pid}")

        profile = InvestorProfile.from_dict(payload)
        self._custom[pid] = profile
        self._save()
        _logger.info("创建自定义 Profile: %s (%s)", pid, profile.name)
        return profile.to_dict()

    def update_profile(self, profile_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新自定义 Profile。预定义不可修改。"""
        if profile_id in PRESET_PROFILES:
            raise ValueError(f"预定义 Profile 不可修改: {profile_id}")
        if profile_id not in self._custom:
            return None

        # 合并更新
        existing = self._custom[profile_id].to_dict()
        existing.update({k: v for k, v in payload.items() if k != "id"})
        profile = InvestorProfile.from_dict(existing)
        self._custom[profile_id] = profile
        self._save()
        _logger.info("更新自定义 Profile: %s", profile_id)
        return profile.to_dict()

    def delete_profile(self, profile_id: str) -> bool:
        """删除自定义 Profile。预定义不可删除。"""
        if profile_id in PRESET_PROFILES:
            raise ValueError(f"预定义 Profile 不可删除: {profile_id}")
        if profile_id not in self._custom:
            return False
        del self._custom[profile_id]
        self._save()
        _logger.info("删除自定义 Profile: %s", profile_id)
        return True

    def apply_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """应用 Profile 到当前会话（返回参数映射）。

        返回 Profile 的完整参数字典，供调用方（前端/回测引擎）消费。
        """
        all_profiles = {**PRESET_PROFILES, **self._custom}
        p = all_profiles.get(profile_id)
        if p is None:
            return None
        return p.to_dict()
