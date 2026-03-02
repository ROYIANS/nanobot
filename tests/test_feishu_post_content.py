import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.feishu import FeishuChannel, _extract_post_content
from nanobot.config.schema import FeishuConfig


def test_extract_post_content_supports_post_wrapper_shape() -> None:
    payload = {
        "post": {
            "zh_cn": {
                "title": "日报",
                "content": [
                    [
                        {"tag": "text", "text": "完成"},
                        {"tag": "img", "image_key": "img_1"},
                    ]
                ],
            }
        }
    }

    text, image_keys = _extract_post_content(payload)

    assert text == "日报 完成"
    assert image_keys == ["img_1"]


def test_extract_post_content_keeps_direct_shape_behavior() -> None:
    payload = {
        "title": "Daily",
        "content": [
            [
                {"tag": "text", "text": "report"},
                {"tag": "img", "image_key": "img_a"},
                {"tag": "img", "image_key": "img_b"},
            ]
        ],
    }

    text, image_keys = _extract_post_content(payload)

    assert text == "Daily report"
    assert image_keys == ["img_a", "img_b"]


def _make_channel() -> FeishuChannel:
    cfg = FeishuConfig(
        enabled=True,
        app_id="cli_test",
        app_secret="secret",
        allow_from=["*"],
    )
    channel = FeishuChannel(cfg, MessageBus())
    channel._client = object()  # bypass uninitialized client guard in send()
    return channel


@pytest.mark.asyncio
async def test_send_uses_text_message_type(monkeypatch) -> None:
    channel = _make_channel()
    sent_calls: list[tuple[str, str, str, str]] = []

    def _fake_send(receive_id_type: str, receive_id: str, msg_type: str, content: str) -> bool:
        sent_calls.append((receive_id_type, receive_id, msg_type, content))
        return True

    monkeypatch.setattr(channel, "_send_message_sync", _fake_send)

    await channel.send(
        OutboundMessage(
            channel="feishu",
            chat_id="ou_123",
            content="hello",
        )
    )

    assert len(sent_calls) == 1
    assert sent_calls[0][2] == "text"
    assert json.loads(sent_calls[0][3]) == {"text": "hello"}


@pytest.mark.asyncio
async def test_on_message_records_reaction_id_for_cleanup(monkeypatch) -> None:
    channel = _make_channel()
    channel._add_reaction = AsyncMock(return_value="reaction_1")
    channel._handle_message = AsyncMock()

    data = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                message_id="om_1",
                chat_id="oc_group",
                chat_type="group",
                message_type="text",
                content=json.dumps({"text": "ping"}),
            ),
            sender=SimpleNamespace(
                sender_type="user",
                sender_id=SimpleNamespace(open_id="ou_user"),
            ),
        )
    )

    await channel._on_message(data)

    assert channel._reaction_ids.get("om_1") == "reaction_1"


@pytest.mark.asyncio
async def test_send_deletes_reaction_after_final_reply(monkeypatch) -> None:
    channel = _make_channel()
    channel._reaction_ids["om_2"] = "reaction_2"

    deleted: list[tuple[str, str]] = []

    def _fake_send(*_args, **_kwargs) -> bool:
        return True

    async def _fake_delete(message_id: str, reaction_id: str) -> None:
        deleted.append((message_id, reaction_id))

    monkeypatch.setattr(channel, "_send_message_sync", _fake_send)
    monkeypatch.setattr(channel, "_delete_reaction", _fake_delete)

    await channel.send(
        OutboundMessage(
            channel="feishu",
            chat_id="ou_123",
            content="done",
            metadata={"message_id": "om_2", "_turn_done": True},
        )
    )

    assert deleted == [("om_2", "reaction_2")]
    assert "om_2" not in channel._reaction_ids


@pytest.mark.asyncio
async def test_send_does_not_delete_reaction_for_progress(monkeypatch) -> None:
    channel = _make_channel()
    channel._reaction_ids["om_3"] = "reaction_3"

    deleted: list[tuple[str, str]] = []

    def _fake_send(*_args, **_kwargs) -> bool:
        return True

    async def _fake_delete(message_id: str, reaction_id: str) -> None:
        deleted.append((message_id, reaction_id))

    monkeypatch.setattr(channel, "_send_message_sync", _fake_send)
    monkeypatch.setattr(channel, "_delete_reaction", _fake_delete)

    await channel.send(
        OutboundMessage(
            channel="feishu",
            chat_id="ou_123",
            content="thinking...",
            metadata={"message_id": "om_3", "_progress": True},
        )
    )

    assert deleted == []
    assert channel._reaction_ids.get("om_3") == "reaction_3"


@pytest.mark.asyncio
async def test_send_does_not_delete_reaction_without_turn_done(monkeypatch) -> None:
    channel = _make_channel()
    channel._reaction_ids["om_4"] = "reaction_4"

    deleted: list[tuple[str, str]] = []

    def _fake_send(*_args, **_kwargs) -> bool:
        return True

    async def _fake_delete(message_id: str, reaction_id: str) -> None:
        deleted.append((message_id, reaction_id))

    monkeypatch.setattr(channel, "_send_message_sync", _fake_send)
    monkeypatch.setattr(channel, "_delete_reaction", _fake_delete)

    await channel.send(
        OutboundMessage(
            channel="feishu",
            chat_id="ou_123",
            content="intermediate message",
            metadata={"message_id": "om_4"},
        )
    )

    assert deleted == []
    assert channel._reaction_ids.get("om_4") == "reaction_4"
