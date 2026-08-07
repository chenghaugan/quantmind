"""Agent 会话记忆测试。"""
import pytest
import tempfile
from pathlib import Path

from quantmind.ai.memory import (
    AgentMemory,
    MemoryMessage,
    MemorySession,
    get_agent_memory,
)


class TestAgentMemory:
    """Agent 记忆管理器测试。"""

    def test_create_session(self):
        """创建会话。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            memory = AgentMemory(db_path)

            session_id = memory.create_session("测试动量因子研究")
            assert session_id.startswith("sess_")
            assert len(session_id) > 10

    def test_add_and_get_message(self):
        """添加和获取消息。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            memory = AgentMemory(db_path)

            session_id = memory.create_session("测试想法")
            memory.add_message(session_id, "user", "研究动量因子")
            memory.add_message(session_id, "assistant", "好的，我来分析动量因子")

            session = memory.get_session(session_id)
            assert session is not None
            assert session.session_id == session_id
            assert session.idea == "测试想法"
            assert len(session.messages) == 2
            assert session.messages[0].role == "user"
            assert session.messages[1].role == "assistant"

    def test_get_nonexistent_session(self):
        """获取不存在的会话返回 None。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            memory = AgentMemory(db_path)

            session = memory.get_session("nonexistent")
            assert session is None

    def test_search_sessions(self):
        """按 idea 检索会话。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            memory = AgentMemory(db_path)

            memory.create_session("动量因子研究")
            memory.create_session("均值回复策略")
            memory.create_session("波动率因子分析")

            results = memory.search_sessions("因子")
            assert len(results) == 2
            ideas = [r["idea"] for r in results]
            assert "动量因子研究" in ideas
            assert "波动率因子分析" in ideas

    def test_list_sessions(self):
        """列出最近会话。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            memory = AgentMemory(db_path)

            memory.create_session("想法1")
            memory.create_session("想法2")
            memory.create_session("想法3")

            sessions = memory.list_sessions(limit=10)
            assert len(sessions) == 3
            # 应该按创建时间倒序
            assert sessions[0]["idea"] == "想法3"
            assert sessions[2]["idea"] == "想法1"

    def test_delete_session(self):
        """删除会话。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            memory = AgentMemory(db_path)

            session_id = memory.create_session("待删除")
            memory.add_message(session_id, "user", "测试消息")

            # 删除
            deleted = memory.delete_session(session_id)
            assert deleted is True

            # 验证已删除
            session = memory.get_session(session_id)
            assert session is None

    def test_delete_nonexistent_session(self):
        """删除不存在的会话返回 False。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            memory = AgentMemory(db_path)

            deleted = memory.delete_session("nonexistent")
            assert deleted is False

    def test_session_with_metadata(self):
        """会话携带元数据。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            memory = AgentMemory(db_path)

            metadata = {"asset_class": "期货", "risk_level": "high"}
            session_id = memory.create_session("测试想法", metadata=metadata)

            session = memory.get_session(session_id)
            assert session is not None
            assert session.metadata == metadata

    def test_message_with_metadata(self):
        """消息携带元数据。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            memory = AgentMemory(db_path)

            session_id = memory.create_session("测试想法")
            memory.add_message(
                session_id,
                "assistant",
                "分析结果",
                metadata={"ic": 0.05, "ir": 1.2},
            )

            session = memory.get_session(session_id)
            assert session is not None
            assert len(session.messages) == 1
            assert session.messages[0].metadata == {"ic": 0.05, "ir": 1.2}


class TestMemoryMessage:
    """消息数据类测试。"""

    def test_to_dict(self):
        """消息序列化为字典。"""
        msg = MemoryMessage(
            role="user",
            content="测试内容",
            timestamp=1234567890.0,
            metadata={"key": "value"},
        )
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "测试内容"
        assert d["timestamp"] == 1234567890.0
        assert d["metadata"] == {"key": "value"}


class TestMemorySession:
    """会话数据类测试。"""

    def test_to_dict(self):
        """会话序列化为字典。"""
        messages = [
            MemoryMessage(role="user", content="问题", timestamp=1.0),
            MemoryMessage(role="assistant", content="回答", timestamp=2.0),
        ]
        session = MemorySession(
            session_id="sess_123",
            idea="测试想法",
            created_at=1234567890.0,
            messages=messages,
            metadata={"status": "complete"},
        )
        d = session.to_dict()
        assert d["session_id"] == "sess_123"
        assert d["idea"] == "测试想法"
        assert len(d["messages"]) == 2
        assert d["metadata"] == {"status": "complete"}
