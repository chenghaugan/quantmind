"""LLM Provider 测试。"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from quantmind.ai.provider import (
    LLMProvider,
    MockProvider,
    _RealProvider,
    build_provider,
)


class TestMockProvider:
    """MockProvider 测试。"""

    @pytest.mark.asyncio
    async def test_mock_research(self):
        """Mock 研究输出结构正确。"""
        provider = MockProvider()
        result = await provider.chat("", "研究一下动量因子")
        assert "asset_class" in result
        assert "hypothesis" in result
        assert "suggested_factors" in result

    @pytest.mark.asyncio
    async def test_mock_factor(self):
        """Mock 因子输出结构正确。"""
        provider = MockProvider()
        result = await provider.chat("", "生成一些因子")
        assert "factors" in result

    @pytest.mark.asyncio
    async def test_mock_code(self):
        """Mock 代码输出包含策略类。"""
        provider = MockProvider()
        result = await provider.chat("", "生成策略代码")
        assert "class" in result
        assert "Strategy" in result


class TestRealProvider:
    """_RealProvider 测试。"""

    @pytest.mark.asyncio
    async def test_real_provider_success(self):
        """真实 Provider 成功调用。"""
        provider = _RealProvider(api_key="test-key", base_url="https://api.test.com/v1")
        
        # Mock httpx 响应
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "测试响应"}}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            result = await provider.chat("system", "user")
            assert result == "测试响应"

    @pytest.mark.asyncio
    async def test_real_provider_network_fallback(self):
        """真实 Provider 网络失败时回退到 Mock。"""
        provider = _RealProvider(api_key="test-key", base_url="https://api.test.com/v1")
        
        # Mock httpx 抛出异常
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Network error")
            )
            result = await provider.chat("", "研究一下动量因子")
            # 应该回退到 Mock，返回结构化 JSON
            assert "asset_class" in result or "factors" in result

    @pytest.mark.asyncio
    async def test_real_provider_timeout_fallback(self):
        """真实 Provider 超时时回退到 Mock。"""
        provider = _RealProvider(api_key="test-key", timeout=0.001)
        
        # Mock httpx 抛出超时
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.ReadTimeout("Timeout")
            )
            result = await provider.chat("", "生成因子")
            # 应该回退到 Mock
            assert "factors" in result


class TestBuildProvider:
    """build_provider 工厂函数测试。"""

    def test_build_mock_by_default(self):
        """默认构建 MockProvider。"""
        provider = build_provider()
        assert isinstance(provider, MockProvider)

    def test_build_mock_without_key(self):
        """无 API key 时构建 MockProvider。"""
        provider = build_provider(name="openai", api_key="")
        assert isinstance(provider, MockProvider)

    def test_build_real_with_key(self):
        """有 API key 时构建 _RealProvider。"""
        provider = build_provider(name="openai", api_key="test-key")
        assert isinstance(provider, _RealProvider)

    def test_build_real_with_custom_config(self):
        """自定义配置构建 _RealProvider。"""
        provider = build_provider(
            name="openai",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            temperature=0.5,
        )
        assert isinstance(provider, _RealProvider)
        assert provider.base_url == "https://api.deepseek.com/v1"
        assert provider.model == "deepseek-chat"
        assert provider.temperature == 0.5
