"""Test message tool final-reply behavior."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ChannelsConfig, FeishuConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path: Path, channels_config: ChannelsConfig | None = None) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        memory_window=10,
        channels_config=channels_config,
    )


class TestMessageToolFinalReplyLogic:
    """Final reply behavior when message tool is used in the same turn."""

    @pytest.mark.asyncio
    async def test_suppress_final_reply_when_sent_to_same_target_and_completed(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        tool_call = ToolCallRequest(
            id="call1", name="message",
            arguments={"content": "Hello", "channel": "feishu", "chat_id": "chat123"},
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="Done", tool_calls=[]),
        ])
        loop.provider.chat = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        sent: list[OutboundMessage] = []
        mt = loop.tools.get("message")
        if isinstance(mt, MessageTool):
            mt.set_send_callback(AsyncMock(side_effect=lambda m: sent.append(m)))

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Send")
        result = await loop._process_message(msg)

        assert len(sent) == 1
        assert result is None

    @pytest.mark.asyncio
    async def test_keep_final_reply_when_sent_to_same_target_but_in_progress(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        tool_call = ToolCallRequest(
            id="call1", name="message",
            arguments={"content": "Quick ack", "channel": "feishu", "chat_id": "chat123"},
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="I'll check the lark_oapi details next.", tool_calls=[]),
        ])
        loop.provider.chat = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        sent: list[OutboundMessage] = []
        mt = loop.tools.get("message")
        if isinstance(mt, MessageTool):
            mt.set_send_callback(AsyncMock(side_effect=lambda m: sent.append(m)))

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Send")
        result = await loop._process_message(msg)

        assert len(sent) == 1
        assert result is not None
        assert "check" in result.content.lower()

    @pytest.mark.asyncio
    async def test_not_suppress_when_sent_to_different_target(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        tool_call = ToolCallRequest(
            id="call1", name="message",
            arguments={"content": "Email content", "channel": "email", "chat_id": "user@example.com"},
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="I've sent the email.", tool_calls=[]),
        ])
        loop.provider.chat = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        sent: list[OutboundMessage] = []
        mt = loop.tools.get("message")
        if isinstance(mt, MessageTool):
            mt.set_send_callback(AsyncMock(side_effect=lambda m: sent.append(m)))

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Send email")
        result = await loop._process_message(msg)

        assert len(sent) == 1
        assert sent[0].channel == "email"
        assert result is not None
        assert result.channel == "feishu"

    @pytest.mark.asyncio
    async def test_not_suppress_when_no_message_tool_used(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        loop.provider.chat = AsyncMock(return_value=LLMResponse(content="Hello!", tool_calls=[]))
        loop.tools.get_definitions = MagicMock(return_value=[])

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Hi")
        result = await loop._process_message(msg)

        assert result is not None
        assert "Hello" in result.content


class TestMessageToolTurnTracking:

    def test_sent_in_turn_tracks_same_target(self) -> None:
        tool = MessageTool()
        tool.set_context("feishu", "chat1")
        assert not tool._sent_in_turn
        tool._sent_in_turn = True
        assert tool._sent_in_turn

    def test_start_turn_resets(self) -> None:
        tool = MessageTool()
        tool._sent_in_turn = True
        tool.start_turn()
        assert not tool._sent_in_turn


class TestFeishuTurnDoneMetadata:
    @pytest.mark.asyncio
    async def test_dispatch_marks_turn_done_on_final_response(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        msg = InboundMessage(
            channel="feishu",
            sender_id="user1",
            chat_id="chat123",
            content="Hi",
            metadata={"message_id": "om_1"},
        )
        loop._process_message = AsyncMock(
            return_value=OutboundMessage(
                channel="feishu",
                chat_id="chat123",
                content="Done",
                metadata={},
            )
        )

        await loop._dispatch(msg)
        out = await loop.bus.consume_outbound()

        assert out.metadata["message_id"] == "om_1"
        assert out.metadata["_turn_done"] is True

    @pytest.mark.asyncio
    async def test_dispatch_emits_turn_done_signal_when_response_suppressed(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        msg = InboundMessage(
            channel="feishu",
            sender_id="user1",
            chat_id="chat123",
            content="Hi",
            metadata={"message_id": "om_2"},
        )
        loop._process_message = AsyncMock(return_value=None)

        await loop._dispatch(msg)
        out = await loop.bus.consume_outbound()

        assert out.content == ""
        assert out.metadata["message_id"] == "om_2"
        assert out.metadata["_turn_done"] is True


class TestFeishuNewSessionSystemMessage:
    @pytest.mark.asyncio
    async def test_new_returns_feishu_system_divider_payload(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)

        async def _ok_consolidate(_session, archive_all: bool = False) -> bool:
            return True

        loop._consolidate_memory = _ok_consolidate  # type: ignore[method-assign]

        msg = InboundMessage(
            channel="feishu",
            sender_id="user1",
            chat_id="chat123",
            content="/new",
            metadata={"message_id": "om_new_1"},
        )
        response = await loop._process_message(msg)

        assert response is not None
        assert response.content == ""
        assert response.metadata["feishu_msg_type"] == "system"
        payload = response.metadata["feishu_system_content"]
        assert payload["type"] == "divider"
        assert payload["params"]["divider_text"]["i18n_text"]["en_US"] == "New Session"


class TestTaskContinueCommand:
    @pytest.mark.asyncio
    async def test_continue_returns_hint_when_no_active_task(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)

        msg = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/continue")
        out = await loop._process_message(msg)

        assert out is not None
        assert "No active task to continue" in out.content

    @pytest.mark.asyncio
    async def test_continue_uses_previous_objective_when_task_in_progress(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        seen_user_messages: list[str] = []
        calls = iter([
            LLMResponse(content="I will check lark_oapi definitions first.", tool_calls=[]),
            LLMResponse(content="Fix completed.", tool_calls=[]),
        ])

        async def _chat(*, messages, **kwargs):
            seen_user_messages.append(messages[-1]["content"])
            return next(calls)

        loop.provider.chat = AsyncMock(side_effect=_chat)
        loop.tools.get_definitions = MagicMock(return_value=[])

        first = await loop._process_message(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="Please fix this bug")
        )
        assert first is not None

        session = loop.sessions.get_or_create("feishu:c1")
        task = session.metadata.get("active_task", {})
        assert task.get("status") == "in_progress"
        assert task.get("objective") == "Please fix this bug"

        second = await loop._process_message(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/continue")
        )
        assert second is not None
        assert second.content == "Fix completed."
        assert "Continue the active task" in seen_user_messages[-1]
        assert "Please fix this bug" in seen_user_messages[-1]

    @pytest.mark.asyncio
    async def test_help_includes_continue_command(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)

        out = await loop._process_message(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/help")
        )

        assert out is not None
        assert "/continue" in out.content

class TestFeishuGroupAndAdminControls:
    @pytest.mark.asyncio
    async def test_group_message_includes_speaker_marker_in_prompt(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        seen_user_inputs: list[str] = []

        async def _chat(*, messages, **kwargs):
            seen_user_inputs.append(messages[-1]["content"])
            return LLMResponse(content="ok", tool_calls=[])

        loop.provider.chat = AsyncMock(side_effect=_chat)
        loop.tools.get_definitions = MagicMock(return_value=[])

        out = await loop._process_message(
            InboundMessage(
                channel="feishu",
                sender_id="ou_user",
                chat_id="oc_group",
                content="please check this",
                metadata={"is_group": True, "group_sender_id": "ou_user"},
            )
        )

        assert out is not None
        assert seen_user_inputs
        assert seen_user_inputs[-1].startswith("[Group speaker: ou_user]")

    @pytest.mark.asyncio
    async def test_non_admin_feishu_sender_cannot_use_admin_tools(self, tmp_path: Path) -> None:
        channels_cfg = ChannelsConfig(
            feishu=FeishuConfig(
                enabled=True,
                app_id="cli_xxx",
                app_secret="secret",
                allow_from=["*"],
                admin_ids=["ou_admin"],
            )
        )
        loop = _make_loop(tmp_path, channels_config=channels_cfg)
        seen_tools: list[set[str]] = []

        async def _chat(*, messages, tools, **kwargs):
            seen_tools.append({t["function"]["name"] for t in tools})
            return LLMResponse(content="ok", tool_calls=[])

        loop.provider.chat = AsyncMock(side_effect=_chat)

        out = await loop._process_message(
            InboundMessage(channel="feishu", sender_id="ou_user", chat_id="c1", content="hello")
        )

        assert out is not None
        assert seen_tools
        names = seen_tools[-1]
        assert "exec" not in names
        assert "edit_file" not in names
        assert "write_file" not in names
        assert "spawn" not in names

    @pytest.mark.asyncio
    async def test_admin_feishu_sender_keeps_admin_tools(self, tmp_path: Path) -> None:
        channels_cfg = ChannelsConfig(
            feishu=FeishuConfig(
                enabled=True,
                app_id="cli_xxx",
                app_secret="secret",
                allow_from=["*"],
                admin_ids=["ou_admin"],
            )
        )
        loop = _make_loop(tmp_path, channels_config=channels_cfg)
        seen_tools: list[set[str]] = []

        async def _chat(*, messages, tools, **kwargs):
            seen_tools.append({t["function"]["name"] for t in tools})
            return LLMResponse(content="ok", tool_calls=[])

        loop.provider.chat = AsyncMock(side_effect=_chat)

        out = await loop._process_message(
            InboundMessage(channel="feishu", sender_id="ou_admin", chat_id="c1", content="hello")
        )

        assert out is not None
        assert seen_tools
        names = seen_tools[-1]
        assert "exec" in names
        assert "edit_file" in names
        assert "write_file" in names

