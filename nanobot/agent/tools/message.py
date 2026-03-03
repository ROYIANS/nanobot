"""Message tool for sending messages to users."""

import json
from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""
    _FEISHU_MSG_TYPES = {
        "text",
        "post",
        "interactive",
        "image",
        "file",
        "audio",
        "media",
        "sticker",
        "share_chat",
        "share_user",
        "system",
    }

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        self._sent_in_turn: bool = False

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "msg_type": {
                    "type": "string",
                    "description": "Optional: message type (for Feishu supports text/post/interactive/system/share_chat/share_user/audio/media/file/sticker/image)"
                },
                "feishu_content": {
                    "description": "Optional: Feishu content JSON object/string for msg_type. For text, defaults to {\"text\": content}.",
                    "oneOf": [
                        {"type": "object"},
                        {"type": "string"}
                    ]
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        msg_type: str | None = None,
        feishu_content: dict[str, Any] | str | None = None,
        media: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id
        message_id = message_id or self._default_message_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        metadata: dict[str, Any] = {
            "message_id": message_id,
        }
        if channel == "feishu" and msg_type:
            normalized = msg_type.strip().lower()
            if normalized not in self._FEISHU_MSG_TYPES:
                supported = ", ".join(sorted(self._FEISHU_MSG_TYPES))
                return f"Error: unsupported feishu msg_type='{normalized}'. Supported: {supported}"
            metadata["feishu_msg_type"] = normalized

            payload = feishu_content
            if payload is None and normalized == "system" and kwargs.get("feishu_system_content") is not None:
                payload = kwargs["feishu_system_content"]

            if normalized == "text":
                if payload is None:
                    payload = {"text": content}
            elif payload is None:
                return f"Error: feishu_content is required when msg_type='{normalized}'"

            if payload is not None:
                payload, err = self._validate_feishu_payload(normalized, payload)
                if err:
                    return err

            if payload is not None:
                metadata["feishu_content"] = payload
                if normalized == "system":
                    metadata["feishu_system_content"] = payload

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata=metadata,
        )

        try:
            await self._send_callback(msg)
            if channel == self._default_channel and chat_id == self._default_chat_id:
                self._sent_in_turn = True
            media_info = f" with {len(media)} attachments" if media else ""
            return f"Message sent to {channel}:{chat_id}{media_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"

    @staticmethod
    def _as_dict_payload(payload: dict[str, Any] | str, msg_type: str) -> tuple[dict[str, Any] | None, str | None]:
        if isinstance(payload, dict):
            return payload, None
        try:
            parsed = json.loads(payload)
        except Exception:
            return None, f"Error: feishu_content for msg_type='{msg_type}' must be a JSON object or JSON string"
        if not isinstance(parsed, dict):
            return None, f"Error: feishu_content for msg_type='{msg_type}' must decode to a JSON object"
        return parsed, None

    def _validate_feishu_payload(
        self,
        msg_type: str,
        payload: dict[str, Any] | str,
    ) -> tuple[dict[str, Any] | str, str | None]:
        obj, err = self._as_dict_payload(payload, msg_type)
        if err:
            return payload, err
        assert obj is not None

        required_fields = {
            "image": ("image_key",),
            "file": ("file_key",),
            "audio": ("file_key",),
            "sticker": ("file_key",),
            "media": ("file_key",),
            "share_chat": ("chat_id",),
            "share_user": ("user_id",),
        }
        if msg_type in required_fields:
            for key in required_fields[msg_type]:
                if not isinstance(obj.get(key), str) or not obj.get(key):
                    return payload, f"Error: feishu_content for msg_type='{msg_type}' requires non-empty '{key}'"
            return obj, None

        if msg_type == "text":
            if not isinstance(obj.get("text"), str) or not obj.get("text"):
                return payload, "Error: feishu_content for msg_type='text' requires non-empty 'text'"
            return obj, None

        if msg_type == "post":
            if not any(isinstance(obj.get(locale), dict) for locale in ("zh_cn", "en_us", "ja_jp", "zh_hk", "zh_tw")):
                return payload, "Error: feishu_content for msg_type='post' requires at least one locale block (e.g. 'zh_cn')"
            return obj, None

        if msg_type == "interactive":
            return obj, None

        if msg_type == "system":
            if obj.get("type") != "divider":
                return payload, "Error: feishu_content for msg_type='system' currently supports type='divider' only"
            divider_text = ((obj.get("params") or {}).get("divider_text") or {})
            if not isinstance(divider_text.get("text"), str) or not divider_text.get("text"):
                return payload, "Error: feishu_content for msg_type='system' requires params.divider_text.text"
            return obj, None

        return obj, None
