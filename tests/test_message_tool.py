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
