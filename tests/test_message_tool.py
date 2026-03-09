import pytest
from unittest.mock import AsyncMock

from nanobot.agent.tools.message import MessageTool


@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"


@pytest.mark.asyncio
async def test_message_tool_passes_feishu_msg_type_and_payload() -> None:
    sent = []
    tool = MessageTool(send_callback=AsyncMock(side_effect=lambda m: sent.append(m)))

    result = await tool.execute(
        content="ignored",
        channel="feishu",
        chat_id="ou_1",
        msg_type="post",
        feishu_content={"zh_cn": {"title": "T", "content": [[{"tag": "text", "text": "hello"}]]}},
    )

    assert "Message sent to feishu:ou_1" in result
    assert len(sent) == 1
    assert sent[0].metadata["feishu_msg_type"] == "post"
    assert sent[0].metadata["feishu_content"]["zh_cn"]["title"] == "T"


@pytest.mark.asyncio
async def test_message_tool_requires_payload_for_non_text_feishu_msg_type() -> None:
    tool = MessageTool(send_callback=AsyncMock())

    result = await tool.execute(
        content="ignored",
        channel="feishu",
        chat_id="ou_1",
        msg_type="interactive",
    )

    assert result == "Error: feishu_content is required when msg_type='interactive'"


@pytest.mark.asyncio
async def test_message_tool_rejects_unknown_feishu_msg_type() -> None:
    tool = MessageTool(send_callback=AsyncMock())

    result = await tool.execute(
        content="ignored",
        channel="feishu",
        chat_id="ou_1",
        msg_type="unknown_type",
        feishu_content={},
    )

    assert "unsupported feishu msg_type='unknown_type'" in result


@pytest.mark.asyncio
async def test_message_tool_validates_required_field_for_share_user() -> None:
    tool = MessageTool(send_callback=AsyncMock())

    result = await tool.execute(
        content="ignored",
        channel="feishu",
        chat_id="ou_1",
        msg_type="share_user",
        feishu_content={},
    )

    assert "requires non-empty 'user_id'" in result


@pytest.mark.asyncio
async def test_message_tool_supports_sticker_file_key_shortcut_and_disables_reply_quote() -> None:
    sent = []
    tool = MessageTool(send_callback=AsyncMock(side_effect=lambda m: sent.append(m)))

    result = await tool.execute(
        content="",
        channel="feishu",
        chat_id="oc_group",
        sticker_file_key="file_v2_sticker_xxx",
    )

    assert "Message sent to feishu:oc_group" in result
    assert len(sent) == 1
    assert sent[0].metadata["feishu_msg_type"] == "sticker"
    assert sent[0].metadata["feishu_content"]["file_key"] == "file_v2_sticker_xxx"
    assert sent[0].metadata["feishu_disable_reply_quote"] is True


@pytest.mark.asyncio
async def test_message_tool_supports_use_recent_sticker(monkeypatch) -> None:
    sent = []
    tool = MessageTool(send_callback=AsyncMock(side_effect=lambda m: sent.append(m)))
    monkeypatch.setattr(
        "nanobot.channels.feishu_sticker_store.latest_chat_sticker",
        lambda chat_id: {"file_key": "file_v2_recent_xxx"} if chat_id == "oc_group" else None,
    )

    result = await tool.execute(
        content="",
        channel="feishu",
        chat_id="oc_group",
        use_recent_sticker=True,
    )

    assert "Message sent to feishu:oc_group" in result
    assert len(sent) == 1
    assert sent[0].metadata["feishu_msg_type"] == "sticker"
    assert sent[0].metadata["feishu_content"]["file_key"] == "file_v2_recent_xxx"


@pytest.mark.asyncio
async def test_message_tool_rejects_same_target_plain_text_reply() -> None:
    sent = []
    tool = MessageTool(send_callback=AsyncMock(side_effect=lambda m: sent.append(m)))
    tool.set_context("feishu", "ou_1")

    result = await tool.execute(
        content="你好呀圈圈",
        channel="feishu",
        chat_id="ou_1",
    )

    assert "Use a normal assistant response instead" in result
    assert sent == []


@pytest.mark.asyncio
async def test_message_tool_allows_same_target_non_text_feishu_message() -> None:
    sent = []
    tool = MessageTool(send_callback=AsyncMock(side_effect=lambda m: sent.append(m)))
    tool.set_context("feishu", "ou_1")

    result = await tool.execute(
        content="",
        channel="feishu",
        chat_id="ou_1",
        sticker_file_key="file_v2_sticker_same_target",
    )

    assert "Message sent to feishu:ou_1" in result
    assert len(sent) == 1
