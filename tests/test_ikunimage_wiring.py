from pathlib import Path

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ExecToolConfig, ProviderConfig
from nanobot.providers.base import LLMProvider, LLMResponse


class _DummyProvider(LLMProvider):
    async def chat(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "gpt-4.1-mini"


def test_agent_loop_registers_ikun_image_tool_when_ikuncode_is_configured(tmp_path: Path):
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_DummyProvider(),
        workspace=tmp_path,
        exec_config=ExecToolConfig(),
        ikuncode_config=ProviderConfig(api_key="sk-ikun"),
    )

    assert loop.tools.has("ikun_image")


@pytest.mark.asyncio
async def test_subagent_manager_registers_ikun_image_tool_when_ikuncode_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = SubagentManager(
        provider=_DummyProvider(),
        workspace=tmp_path,
        bus=MessageBus(),
        exec_config=ExecToolConfig(),
        ikuncode_config=ProviderConfig(api_key="sk-ikun"),
    )

    registered: list[str] = []
    original_register = ToolRegistry.register

    def _register(self, tool):
        registered.append(tool.name)
        return original_register(self, tool)

    async def _announce_result(*args, **kwargs):
        return None

    monkeypatch.setattr(ToolRegistry, "register", _register)
    monkeypatch.setattr(manager, "_announce_result", _announce_result)

    await manager._run_subagent("task-1", "do something", "label", {"channel": "cli", "chat_id": "direct"})

    assert "ikun_image" in registered
