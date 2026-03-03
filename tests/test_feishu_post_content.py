import json
import sys
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
        group_policy="open",
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
    create_card = AsyncMock()
    monkeypatch.setattr(channel, "_create_processing_card_for_message", create_card)

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
    create_card.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_message_skips_processing_card_for_slash_command(monkeypatch) -> None:
    channel = _make_channel()
    channel._add_reaction = AsyncMock(return_value="reaction_cmd")
    channel._handle_message = AsyncMock()
    create_card = AsyncMock()
    monkeypatch.setattr(channel, "_create_processing_card_for_message", create_card)

    data = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                message_id="om_cmd_1",
                chat_id="oc_group",
                chat_type="group",
                message_type="text",
                content=json.dumps({"text": "/new"}),
            ),
            sender=SimpleNamespace(
                sender_type="user",
                sender_id=SimpleNamespace(open_id="ou_user"),
            ),
        )
    )

    await channel._on_message(data)

    assert channel._reaction_ids.get("om_cmd_1") == "reaction_cmd"
    create_card.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_message_group_mention_policy_ignores_non_mention(monkeypatch) -> None:
    channel = _make_channel()
    channel.config.group_policy = "mention"
    channel.config.bot_open_id = "ou_bot"
    channel.config.proactive_reply_probability = 0.0
    channel._add_reaction = AsyncMock(return_value="reaction_ignored")
    channel._handle_message = AsyncMock()
    create_card = AsyncMock()
    monkeypatch.setattr(channel, "_create_processing_card_for_message", create_card)

    data = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                message_id="om_group_ignored",
                chat_id="oc_group",
                chat_type="group",
                message_type="text",
                content=json.dumps({"text": "just chatting"}),
            ),
            sender=SimpleNamespace(
                sender_type="user",
                sender_id=SimpleNamespace(open_id="ou_user"),
            ),
        )
    )

    await channel._on_message(data)

    channel._add_reaction.assert_not_awaited()
    create_card.assert_not_awaited()
    channel._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_message_group_mention_policy_replies_when_bot_is_mentioned(monkeypatch) -> None:
    channel = _make_channel()
    channel.config.group_policy = "mention"
    channel.config.bot_open_id = "ou_bot"
    channel._add_reaction = AsyncMock(return_value="reaction_mention")
    channel._handle_message = AsyncMock()
    create_card = AsyncMock()
    monkeypatch.setattr(channel, "_create_processing_card_for_message", create_card)

    data = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                message_id="om_group_mention",
                chat_id="oc_group",
                chat_type="group",
                message_type="text",
                content=json.dumps({"text": "<at user_id=\"ou_bot\">bot</at> ping"}),
            ),
            sender=SimpleNamespace(
                sender_type="user",
                sender_id=SimpleNamespace(open_id="ou_user"),
            ),
        )
    )

    await channel._on_message(data)

    create_card.assert_awaited_once()
    channel._handle_message.assert_awaited_once()
    metadata = channel._handle_message.await_args.kwargs["metadata"]
    assert metadata["is_group"] is True
    assert metadata["group_sender_id"] == "ou_user"
    assert metadata["was_mentioned"] is True
    assert metadata["proactive_reply"] is False


@pytest.mark.asyncio
async def test_on_message_group_mention_policy_can_reply_proactively(monkeypatch) -> None:
    channel = _make_channel()
    channel.config.group_policy = "mention"
    channel.config.bot_open_id = "ou_bot"
    channel.config.proactive_reply_probability = 0.5
    channel._add_reaction = AsyncMock(return_value="reaction_proactive")
    channel._handle_message = AsyncMock()
    create_card = AsyncMock()
    monkeypatch.setattr(channel, "_create_processing_card_for_message", create_card)
    monkeypatch.setattr("nanobot.channels.feishu.random.random", lambda: 0.01)

    data = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                message_id="om_group_proactive",
                chat_id="oc_group",
                chat_type="group",
                message_type="text",
                content=json.dumps({"text": "normal group message"}),
            ),
            sender=SimpleNamespace(
                sender_type="user",
                sender_id=SimpleNamespace(open_id="ou_user"),
            ),
        )
    )

    await channel._on_message(data)

    create_card.assert_awaited_once()
    channel._handle_message.assert_awaited_once()
    metadata = channel._handle_message.await_args.kwargs["metadata"]
    assert metadata["was_mentioned"] is False
    assert metadata["proactive_reply"] is True


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


@pytest.mark.asyncio
async def test_send_progress_updates_card_without_sending_text(monkeypatch) -> None:
    channel = _make_channel()
    channel._processing_cards["om_p"] = "om_card_p"
    channel._processing_card_text["om_p"] = "old"

    sent_calls: list[tuple[str, str, str, str]] = []

    def _fake_send(receive_id_type: str, receive_id: str, msg_type: str, content: str) -> str | None:
        sent_calls.append((receive_id_type, receive_id, msg_type, content))
        return "om_sent"

    update = AsyncMock()
    monkeypatch.setattr(channel, "_send_message_sync", _fake_send)
    monkeypatch.setattr(channel, "_update_processing_card_for_message", update)

    await channel.send(
        OutboundMessage(
            channel="feishu",
            chat_id="ou_123",
            content="thinking...",
            metadata={"message_id": "om_p", "_progress": True},
        )
    )

    update.assert_awaited_once()
    assert sent_calls == []


@pytest.mark.asyncio
async def test_update_processing_card_appends_logs_instead_of_overwrite(monkeypatch) -> None:
    channel = _make_channel()
    channel._processing_cards["om_append"] = "om_card_append"
    channel._processing_card_logs["om_append"] = []
    channel._processing_card_step["om_append"] = 0

    payloads: list[dict] = []

    def _fake_update(_message_id: str, _msg_type: str, content: str) -> bool:
        payloads.append(json.loads(content))
        return True

    monkeypatch.setattr(channel, "_update_message_sync", _fake_update)

    await channel._update_processing_card_for_message("om_append", "处理文件 A")
    await channel._update_processing_card_for_message("om_append", "调用工具 read_file")

    assert channel._processing_card_logs["om_append"] == ["处理文件 A", "调用工具 read_file"]
    assert len(payloads) == 2
    logs_md = payloads[-1]["elements"][-1]["content"]
    assert "> 处理文件 A" in logs_md
    assert "> 调用工具 read_file" in logs_md


@pytest.mark.asyncio
async def test_update_processing_card_uses_dynamic_thinking_dots(monkeypatch) -> None:
    channel = _make_channel()
    channel._processing_cards["om_dot"] = "om_card_dot"
    channel._processing_card_logs["om_dot"] = []
    channel._processing_card_step["om_dot"] = 0

    status_lines: list[str] = []

    def _fake_update(_message_id: str, _msg_type: str, content: str) -> bool:
        payload = json.loads(content)
        status_lines.append(payload["elements"][0]["content"])
        return True

    monkeypatch.setattr(channel, "_update_message_sync", _fake_update)

    await channel._update_processing_card_for_message("om_dot", "first")
    await channel._update_processing_card_for_message("om_dot", "second")

    assert len(status_lines) == 2
    assert status_lines[0].startswith("> 正在思考")
    assert status_lines[1].startswith("> 正在思考")
    assert status_lines[0] != status_lines[1]


@pytest.mark.asyncio
async def test_send_turn_done_cleans_card_and_reaction(monkeypatch) -> None:
    channel = _make_channel()
    channel._processing_cards["om_done"] = "om_card_done"
    channel._reaction_ids["om_done"] = "reaction_done"

    sent_calls: list[tuple[str, str, str, str]] = []

    def _fake_send(receive_id_type: str, receive_id: str, msg_type: str, content: str) -> str | None:
        sent_calls.append((receive_id_type, receive_id, msg_type, content))
        return "om_sent"

    del_card = AsyncMock()
    del_reaction = AsyncMock()
    monkeypatch.setattr(channel, "_send_message_sync", _fake_send)
    monkeypatch.setattr(channel, "_delete_processing_card_for_message", del_card)
    monkeypatch.setattr(channel, "_delete_reaction_for_message", del_reaction)

    await channel.send(
        OutboundMessage(
            channel="feishu",
            chat_id="ou_123",
            content="final",
            metadata={"message_id": "om_done", "_turn_done": True},
        )
    )

    assert sent_calls and sent_calls[0][2] == "text"
    del_card.assert_awaited_once_with("om_done")
    del_reaction.assert_awaited_once_with("om_done")


@pytest.mark.asyncio
async def test_send_system_divider_message_from_metadata(monkeypatch) -> None:
    channel = _make_channel()
    sent_calls: list[tuple[str, str, str, str]] = []

    def _fake_send(receive_id_type: str, receive_id: str, msg_type: str, content: str) -> str | None:
        sent_calls.append((receive_id_type, receive_id, msg_type, content))
        return "om_system"

    monkeypatch.setattr(channel, "_send_message_sync", _fake_send)

    payload = {
        "type": "divider",
        "params": {
            "divider_text": {
                "text": "新会话",
                "i18n_text": {"zh_CN": "新会话", "en_US": "New Session"},
            }
        },
        "options": {"need_rollup": True},
    }
    await channel.send(
        OutboundMessage(
            channel="feishu",
            chat_id="ou_123",
            content="New session started.",
            metadata={
                "feishu_msg_type": "system",
                "feishu_system_content": payload,
            },
        )
    )

    assert len(sent_calls) == 1
    assert sent_calls[0][2] == "system"
    assert json.loads(sent_calls[0][3]) == payload


@pytest.mark.asyncio
async def test_send_post_message_from_force_msg_type_and_payload(monkeypatch) -> None:
    channel = _make_channel()
    sent_calls: list[tuple[str, str, str, str]] = []

    def _fake_send(receive_id_type: str, receive_id: str, msg_type: str, content: str) -> str | None:
        sent_calls.append((receive_id_type, receive_id, msg_type, content))
        return "om_post"

    monkeypatch.setattr(channel, "_send_message_sync", _fake_send)

    payload = {"zh_cn": {"title": "日报", "content": [[{"tag": "text", "text": "完成"}]]}}
    await channel.send(
        OutboundMessage(
            channel="feishu",
            chat_id="ou_123",
            content="fallback text",
            metadata={
                "feishu_msg_type": "post",
                "feishu_content": payload,
            },
        )
    )

    assert len(sent_calls) == 1
    assert sent_calls[0][2] == "post"
    assert json.loads(sent_calls[0][3]) == payload


def test_update_message_sync_supports_patch_body_without_msg_type(monkeypatch) -> None:
    channel = _make_channel()

    recorded: dict[str, object] = {}

    class _BodyBuilder:
        def __init__(self) -> None:
            self._content = None

        def content(self, value: str):
            self._content = value
            return self

        def build(self):
            return {"content": self._content}

    class _Body:
        @staticmethod
        def builder():
            return _BodyBuilder()

    class _ReqBuilder:
        def __init__(self) -> None:
            self._message_id = None
            self._body = None

        def message_id(self, value: str):
            self._message_id = value
            return self

        def request_body(self, value):
            self._body = value
            return self

        def build(self):
            return {"message_id": self._message_id, "body": self._body}

    class _Req:
        @staticmethod
        def builder():
            return _ReqBuilder()

    fake_v1 = SimpleNamespace(
        PatchMessageRequest=_Req,
        PatchMessageRequestBody=_Body,
    )
    monkeypatch.setitem(sys.modules, "lark_oapi.api.im.v1", fake_v1)

    class _FakeMessageApi:
        def patch(self, request):
            recorded["request"] = request
            return SimpleNamespace(success=lambda: True)

    channel._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=_FakeMessageApi()))
    )

    ok = channel._update_message_sync("om_patch", "interactive", "{\"x\":1}")

    assert ok is True
    assert recorded["request"]["message_id"] == "om_patch"
    assert recorded["request"]["body"] == {"content": "{\"x\":1}"}


def test_upload_file_sync_rejects_empty_file(tmp_path) -> None:
    channel = _make_channel()
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")

    assert channel._upload_file_sync(str(p)) is None


def test_upload_file_sync_rejects_file_larger_than_30mb(tmp_path, monkeypatch) -> None:
    channel = _make_channel()
    p = tmp_path / "big.pdf"
    p.write_bytes(b"1")
    monkeypatch.setattr("os.path.getsize", lambda _path: (30 * 1024 * 1024) + 1)

    assert channel._upload_file_sync(str(p)) is None
