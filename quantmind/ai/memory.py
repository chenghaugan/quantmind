"""轻量级 Agent 会话记忆（SQLite 多轮对话）。

对标 Vibe-Trading 的 memory Tier 1/2，实现基于 SQLite 的会话记忆系统，
支持多轮对话上下文检索和历史研究过程回溯。

设计要点：
1. 每次研究会话（session）独立存储
2. 支持按 session_id 检索完整对话历史
3. 支持按 idea 关键词检索相似研究
4. 轻量级，无外部依赖（仅 SQLite）
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_logger = logging.getLogger("quantmind.ai.memory")


@dataclass
class MemoryMessage:
    """会话中的单条消息。"""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {},
        }


@dataclass
class MemorySession:
    """研究会话。"""

    session_id: str
    idea: str
    created_at: float
    messages: List[MemoryMessage]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "idea": self.idea,
            "created_at": self.created_at,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata or {},
        }


class AgentMemory:
    """Agent 会话记忆管理器（SQLite 后端）。"""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            # 默认路径：quantmind/db/agent_memory.db
            root = Path(__file__).resolve().parent.parent.parent
            db_path = str(root / "db" / "agent_memory.db")
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        """SQLite 连接上下文管理器。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """初始化数据库表。"""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    idea TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_idea
                ON sessions(idea)
            """)

    def create_session(self, idea: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """创建新会话，返回 session_id。"""
        session_id = f"sess_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        created_at = time.time()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, idea, created_at, metadata) VALUES (?, ?, ?, ?)",
                (session_id, idea, created_at, metadata_json),
            )
        _logger.info("创建会话: %s, idea=%s", session_id, idea[:50])
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """向会话添加消息。"""
        timestamp = time.time()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, timestamp, metadata_json),
            )
        _logger.debug("添加消息: session=%s, role=%s, len=%d", session_id, role, len(content))

    def get_session(self, session_id: str) -> Optional[MemorySession]:
        """获取完整会话（含所有消息）。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, idea, created_at, metadata FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None

            messages_rows = conn.execute(
                "SELECT role, content, timestamp, metadata FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()

        messages = [
            MemoryMessage(
                role=r["role"],
                content=r["content"],
                timestamp=r["timestamp"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None,
            )
            for r in messages_rows
        ]

        return MemorySession(
            session_id=row["session_id"],
            idea=row["idea"],
            created_at=row["created_at"],
            messages=messages,
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )

    def search_sessions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """按 idea 关键词检索会话（简单子串匹配）。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, idea, created_at, metadata FROM sessions WHERE idea LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()

        return [
            {
                "session_id": r["session_id"],
                "idea": r["idea"],
                "created_at": r["created_at"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
            }
            for r in rows
        ]

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """列出最近会话（不含消息内容）。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, idea, created_at, metadata FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            {
                "session_id": r["session_id"],
                "idea": r["idea"],
                "created_at": r["created_at"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
            }
            for r in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其所有消息。"""
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            result = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            deleted = result.rowcount > 0
        if deleted:
            _logger.info("删除会话: %s", session_id)
        return deleted


# 全局单例（延迟初始化）
_memory: Optional[AgentMemory] = None


def get_agent_memory() -> AgentMemory:
    """获取全局 AgentMemory 单例。"""
    global _memory
    if _memory is None:
        _memory = AgentMemory()
    return _memory


__all__ = [
    "MemoryMessage",
    "MemorySession",
    "AgentMemory",
    "get_agent_memory",
]
