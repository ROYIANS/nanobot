"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import json
import os
import random
import re
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateFileRequest,
        CreateFileRequestBody,
        CreateImageRequest,
        CreateImageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        CreateMessageRequest,
        CreateMessageRequestBody,
        Emoji,
        GetMessageResourceRequest,
        P2ImMessageReceiveV1,
    )
    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None
    Emoji = None

# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}


def _extract_share_card_content(content_json: dict, msg_type: str) -> str:
    """Extract text representation from share cards and interactive messages."""
    parts = []

    if msg_type == "share_chat":
        parts.append(f"[shared chat: {content_json.get('chat_id', '')}]")
    elif msg_type == "share_user":
        parts.append(f"[shared user: {content_json.get('user_id', '')}]")
    elif msg_type == "interactive":
        parts.extend(_extract_interactive_content(content_json))
    elif msg_type == "share_calendar_event":
        parts.append(f"[shared calendar event: {content_json.get('event_key', '')}]")
    elif msg_type == "system":
        parts.append("[system message]")
    elif msg_type == "merge_forward":
        parts.append("[merged forward messages]")

    return "\n".join(parts) if parts else f"[{msg_type}]"


def _extract_interactive_content(content: dict) -> list[str]:
    """Recursively extract text and links from interactive card content."""
    parts = []

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return [content] if content.strip() else []

    if not isinstance(content, dict):
        return parts

    if "title" in content:
        title = content["title"]
        if isinstance(title, dict):
            title_content = title.get("content", "") or title.get("text", "")
            if title_content:
                parts.append(f"title: {title_content}")
        elif isinstance(title, str):
            parts.append(f"title: {title}")

    for elements in content.get("elements", []) if isinstance(content.get("elements"), list) else []:
        for element in elements:
            parts.extend(_extract_element_content(element))

    card = content.get("card", {})
    if card:
        parts.extend(_extract_interactive_content(card))

    header = content.get("header", {})
    if header:
        header_title = header.get("title", {})
        if isinstance(header_title, dict):
            header_text = header_title.get("content", "") or header_title.get("text", "")
            if header_text:
                parts.append(f"title: {header_text}")

    return parts


def _extract_element_content(element: dict) -> list[str]:
    """Extract content from a single card element."""
    parts = []

    if not isinstance(element, dict):
        return parts

    tag = element.get("tag", "")

    if tag in ("markdown", "lark_md"):
        content = element.get("content", "")
        if content:
            parts.append(content)

    elif tag == "div":
        text = element.get("text", {})
        if isinstance(text, dict):
            text_content = text.get("content", "") or text.get("text", "")
            if text_content:
                parts.append(text_content)
        elif isinstance(text, str):
            parts.append(text)
        for field in element.get("fields", []):
            if isinstance(field, dict):
                field_text = field.get("text", {})
                if isinstance(field_text, dict):
                    c = field_text.get("content", "")
                    if c:
                        parts.append(c)

    elif tag == "a":
        href = element.get("href", "")
        text = element.get("text", "")
        if href:
            parts.append(f"link: {href}")
        if text:
            parts.append(text)

    elif tag == "button":
        text = element.get("text", {})
        if isinstance(text, dict):
            c = text.get("content", "")
            if c:
                parts.append(c)
        url = element.get("url", "") or element.get("multi_url", {}).get("url", "")
        if url:
            parts.append(f"link: {url}")

    elif tag == "img":
        alt = element.get("alt", {})
        parts.append(alt.get("content", "[image]") if isinstance(alt, dict) else "[image]")

    elif tag == "note":
        for ne in element.get("elements", []):
            parts.extend(_extract_element_content(ne))

    elif tag == "column_set":
        for col in element.get("columns", []):
            for ce in col.get("elements", []):
                parts.extend(_extract_element_content(ce))

    elif tag == "plain_text":
        content = element.get("content", "")
        if content:
            parts.append(content)

    else:
        for ne in element.get("elements", []):
            parts.extend(_extract_element_content(ne))

    return parts


def _extract_post_content(content_json: dict) -> tuple[str, list[str]]:
    """Extract text and image keys from Feishu post (rich text) message.

    Handles three payload shapes:
    - Direct:    {"title": "...", "content": [[...]]}
    - Localized: {"zh_cn": {"title": "...", "content": [...]}}
    - Wrapped:   {"post": {"zh_cn": {"title": "...", "content": [...]}}}
    """

    def _parse_block(block: dict) -> tuple[str | None, list[str]]:
        if not isinstance(block, dict) or not isinstance(block.get("content"), list):
            return None, []
        texts, images = [], []
        if title := block.get("title"):
            texts.append(title)
        for row in block["content"]:
            if not isinstance(row, list):
                continue
            for el in row:
                if not isinstance(el, dict):
                    continue
                tag = el.get("tag")
                if tag in ("text", "a"):
                    texts.append(el.get("text", ""))
                elif tag == "at":
                    texts.append(f"@{el.get('user_name', 'user')}")
                elif tag == "img" and (key := el.get("image_key")):
                    images.append(key)
        return (" ".join(texts).strip() or None), images

    # Unwrap optional {"post": ...} envelope
    root = content_json
    if isinstance(root, dict) and isinstance(root.get("post"), dict):
        root = root["post"]
    if not isinstance(root, dict):
        return "", []

    # Direct format
    if "content" in root:
        text, imgs = _parse_block(root)
        if text or imgs:
            return text or "", imgs

    # Localized: prefer known locales, then fall back to any dict child
    for key in ("zh_cn", "en_us", "ja_jp"):
        if key in root:
            text, imgs = _parse_block(root[key])
            if text or imgs:
                return text or "", imgs
    for val in root.values():
        if isinstance(val, dict):
            text, imgs = _parse_block(val)
            if text or imgs:
                return text or "", imgs

    return "", []


def _extract_post_text(content_json: dict) -> str:
    """Extract plain text from Feishu post (rich text) message content.

    Legacy wrapper for _extract_post_content, returns only text.
    """
    text, _ = _extract_post_content(content_json)
    return text


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark channel using WebSocket long connection.

    Uses WebSocket to receive events - no public IP or webhook required.

    Requires:
    - App ID and App Secret from Feishu Open Platform
    - Bot capability enabled
    - Event subscription enabled (im.message.receive_v1)
    """

    name = "feishu"
    _MAX_UPLOAD_FILE_BYTES = 30 * 1024 * 1024
    _NON_OPUS_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".amr"}
    _AT_USER_ID_RE = re.compile(r"<at\b[^>]*\buser_id\s*=\s*[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)

    def __init__(self, config: FeishuConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()  # Ordered dedup cache
        self._reaction_ids: OrderedDict[str, str] = OrderedDict()  # message_id -> reaction_id
        self._processing_cards: OrderedDict[str, str] = OrderedDict()  # source message_id -> card message_id
        self._processing_card_text: OrderedDict[str, str] = OrderedDict()  # source message_id -> latest render key
        self._processing_card_logs: OrderedDict[str, list[str]] = OrderedDict()  # source message_id -> appended logs
        self._processing_card_step: OrderedDict[str, int] = OrderedDict()  # source message_id -> dot animation step
        self._processing_card_animators: dict[str, asyncio.Task[None]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._bot_open_id: str = str(config.bot_open_id or "").strip()
        self._user_name_cache: dict[str, str] = {}  # open_id → display name

    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return

        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return

        self._running = True
        self._loop = asyncio.get_running_loop()

        # Create Lark client for sending messages
        self._client = lark.Client.builder() \
            .app_id(self.config.app_id) \
            .app_secret(self.config.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        await self._refresh_bot_open_id_via_api()

        # Create event handler (only register message receive, ignore other events)
        event_handler = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(
            self._on_message_sync
        ).build()

        # Create WebSocket client for long connection
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO
        )

        # Start WebSocket client in a separate thread with reconnect loop
        def run_ws():
            while self._running:
                try:
                    self._ws_client.start()
                except Exception as e:
                    logger.warning("Feishu WebSocket error: {}", e)
                if self._running:
                    import time
                    time.sleep(5)

        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()

        logger.info("Feishu bot started with WebSocket long connection")
        logger.info("No public IP required - using WebSocket to receive events")

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """
        Stop the Feishu bot.

        Notice: lark.ws.Client does not expose stop method， simply exiting the program will close the client.

        Reference: https://github.com/larksuite/oapi-sdk-python/blob/v2_main/lark_oapi/ws/client.py#L86
        """
        self._running = False
        logger.info("Feishu bot stopped")

    def _fetch_bot_open_id_sync(self) -> str | None:
        """Fetch bot open_id from Feishu OpenAPI (/open-apis/bot/v3/info)."""
        if self._bot_open_id:
            return self._bot_open_id
        if not self.config.app_id or not self.config.app_secret:
            return None
        try:
            import requests
        except Exception:
            return None

        try:
            token_resp = requests.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": self.config.app_id, "app_secret": self.config.app_secret},
                timeout=8,
            )
            token_data = token_resp.json() if token_resp is not None else {}
            if token_data.get("code", 0) != 0:
                logger.warning(
                    "Failed to get tenant_access_token for bot id lookup: code={}, msg={}",
                    token_data.get("code"),
                    token_data.get("msg"),
                )
                return None
            token = str(token_data.get("tenant_access_token", "")).strip()
            if not token:
                return None

            info_resp = requests.get(
                "https://open.feishu.cn/open-apis/bot/v3/info",
                headers={"Authorization": f"Bearer {token}"},
                timeout=8,
            )
            info_data = info_resp.json() if info_resp is not None else {}
            if info_data.get("code", 0) != 0:
                logger.warning(
                    "Failed to fetch bot info: code={}, msg={}",
                    info_data.get("code"),
                    info_data.get("msg"),
                )
                return None

            bot_obj = info_data.get("bot")
            if not isinstance(bot_obj, dict):
                data_obj = info_data.get("data")
                if isinstance(data_obj, dict):
                    bot_obj = data_obj.get("bot") if isinstance(data_obj.get("bot"), dict) else data_obj
            if not isinstance(bot_obj, dict):
                return None

            open_id = str(bot_obj.get("open_id", "")).strip()
            if open_id:
                logger.info("Resolved Feishu bot open_id via API: {}", open_id)
                return open_id
            return None
        except Exception as e:
            logger.debug("Bot open_id API lookup failed: {}", e)
            return None

    async def _refresh_bot_open_id_via_api(self, force: bool = False) -> str:
        """Resolve bot open_id via OpenAPI and cache it."""
        if self._bot_open_id and not force:
            return self._bot_open_id
        loop = asyncio.get_running_loop()
        resolved = await loop.run_in_executor(None, self._fetch_bot_open_id_sync)
        if resolved:
            self._bot_open_id = resolved
            self.config.bot_open_id = resolved
        return self._bot_open_id

    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> str | None:
        """Sync helper for adding reaction (runs in thread pool)."""
        try:
            request = CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                ).build()

            response = self._client.im.v1.message_reaction.create(request)

            if not response.success():
                logger.warning("Failed to add reaction: code={}, msg={}", response.code, response.msg)
                return None
            else:
                logger.debug("Added {} reaction to message {}", emoji_type, message_id)
                return getattr(response.data, "reaction_id", None)
        except Exception as e:
            logger.warning("Error adding reaction: {}", e)
            return None

    async def _add_reaction(self, message_id: str, emoji_type: str = "THUMBSUP") -> str | None:
        """
        Add a reaction emoji to a message (non-blocking).

        Common emoji types: THUMBSUP, OK, EYES, DONE, OnIt, HEART
        """
        if not self._client or not Emoji:
            return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._add_reaction_sync, message_id, emoji_type)

    def _delete_reaction_sync(self, message_id: str, reaction_id: str) -> None:
        """Sync helper for deleting a reaction (runs in thread pool)."""
        try:
            try:
                from lark_oapi.api.im.v1 import DeleteMessageReactionRequest
            except ImportError:
                logger.warning("DeleteMessageReactionRequest not available in current lark-oapi version")
                return
            request = DeleteMessageReactionRequest.builder() \
                .message_id(message_id) \
                .reaction_id(reaction_id) \
                .build()
            response = self._client.im.v1.message_reaction.delete(request)
            if not response.success():
                logger.warning(
                    "Failed to delete reaction: message_id={}, reaction_id={}, code={}, msg={}",
                    message_id, reaction_id, response.code, response.msg
                )
            else:
                logger.debug("Deleted reaction {} from message {}", reaction_id, message_id)
        except Exception as e:
            logger.warning("Error deleting reaction {} from message {}: {}", reaction_id, message_id, e)

    async def _delete_reaction(self, message_id: str, reaction_id: str) -> None:
        """Delete a reaction emoji from a message (non-blocking)."""
        if not self._client:
            return
        if not reaction_id:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._delete_reaction_sync, message_id, reaction_id)

    async def _delete_reaction_for_message(self, message_id: str) -> None:
        """Delete and clear tracked reaction for an inbound message."""
        reaction_id = self._reaction_ids.get(message_id)
        if not reaction_id:
            return
        try:
            await self._delete_reaction(message_id, reaction_id)
        finally:
            self._reaction_ids.pop(message_id, None)

    @staticmethod
    def _build_processing_card(progress_text: str, logs: list[str] | None = None) -> dict:
        """Build a lightweight progress card for one turn."""
        status = (progress_text or "").strip() or "正在思考."
        log_lines = [line.strip() for line in (logs or []) if str(line).strip()]
        status_quote = f"> {status}"
        logs_quote = "\n".join(f"> {line}" for line in log_lines) if log_lines else ""

        elements: list[dict[str, Any]] = [{"tag": "markdown", "content": status_quote}]
        if logs_quote:
            elements.append({"tag": "markdown", "content": logs_quote})

        return {
            "config": {"wide_screen_mode": True, "update_multi": True},
            "header": {
                "title": {"tag": "plain_text", "content": "Nanobot 正在处理中"},
                "template": "blue",
            },
            "elements": elements,
        }

    @staticmethod
    def _thinking_text(step: int) -> str:
        dots = "." * ((step % 3) + 1)
        return f"正在思考{dots}"

    def _render_key(self, status: str, logs: list[str]) -> str:
        return status + "\n" + "\n".join(logs)

    async def _render_processing_card(self, source_message_id: str, *, force: bool = False) -> None:
        card_message_id = self._processing_cards.get(source_message_id)
        if not card_message_id:
            return
        logs = list(self._processing_card_logs.get(source_message_id, []))
        step = int(self._processing_card_step.get(source_message_id, 0))
        status = self._thinking_text(step)
        render_key = self._render_key(status, logs)
        if not force and self._processing_card_text.get(source_message_id) == render_key:
            return

        card = self._build_processing_card(status, logs)
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(
            None,
            self._update_message_sync,
            card_message_id,
            "interactive",
            json.dumps(card, ensure_ascii=False),
        )
        if ok:
            self._processing_card_text[source_message_id] = render_key

    async def _animate_processing_card(self, source_message_id: str) -> None:
        """Keep lightweight dot animation while the card is alive."""
        try:
            while source_message_id in self._processing_cards:
                await asyncio.sleep(1.0)
                if source_message_id not in self._processing_cards:
                    break
                self._processing_card_step[source_message_id] = int(
                    self._processing_card_step.get(source_message_id, 0)
                ) + 1
                await self._render_processing_card(source_message_id)
        except asyncio.CancelledError:
            return

    def _update_message_sync(self, message_id: str, msg_type: str, content: str) -> bool:
        """Sync helper for updating an existing message."""
        try:
            try:
                from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody
            except ImportError:
                logger.warning("PatchMessageRequest not available in current lark-oapi version")
                return False
            # Some lark-oapi versions don't provide msg_type() on PatchMessageRequestBodyBuilder.
            # For process-card updates we only need to patch content.
            body = PatchMessageRequestBody.builder().content(content).build()
            request = PatchMessageRequest.builder() \
                .message_id(message_id) \
                .request_body(body).build()
            response = self._client.im.v1.message.patch(request)
            if not response.success():
                logger.warning("Failed to update message: message_id={}, code={}, msg={}", message_id, response.code, response.msg)
                return False
            return True
        except Exception as e:
            logger.warning("Error updating message {}: {}", message_id, e)
            return False

    def _delete_message_sync(self, message_id: str) -> bool:
        """Sync helper for deleting a message (message recall)."""
        try:
            try:
                from lark_oapi.api.im.v1 import DeleteMessageRequest
            except ImportError:
                logger.warning("DeleteMessageRequest not available in current lark-oapi version")
                return False
            request = DeleteMessageRequest.builder().message_id(message_id).build()
            response = self._client.im.v1.message.delete(request)
            if not response.success():
                logger.warning("Failed to delete message: message_id={}, code={}, msg={}", message_id, response.code, response.msg)
                return False
            return True
        except Exception as e:
            logger.warning("Error deleting message {}: {}", message_id, e)
            return False

    async def _create_processing_card_for_message(self, source_message_id: str, receive_id_type: str, receive_id: str) -> None:
        """Create a process card for one inbound message if absent."""
        if not source_message_id or source_message_id in self._processing_cards:
            return
        self._processing_card_logs[source_message_id] = []
        self._processing_card_step[source_message_id] = 0
        status = self._thinking_text(0)
        card = self._build_processing_card(status, [])
        loop = asyncio.get_running_loop()
        card_message_id = await loop.run_in_executor(
            None,
            self._send_message_sync,
            receive_id_type,
            receive_id,
            "interactive",
            json.dumps(card, ensure_ascii=False),
        )
        if card_message_id:
            self._processing_cards[source_message_id] = card_message_id
            self._processing_card_text[source_message_id] = self._render_key(status, [])
            self._processing_card_animators[source_message_id] = asyncio.create_task(
                self._animate_processing_card(source_message_id)
            )
            while len(self._processing_cards) > 1000:
                old_key, _ = self._processing_cards.popitem(last=False)
                self._processing_card_text.pop(old_key, None)
                self._processing_card_logs.pop(old_key, None)
                self._processing_card_step.pop(old_key, None)
                animator = self._processing_card_animators.pop(old_key, None)
                if animator:
                    animator.cancel()

    async def _update_processing_card_for_message(self, source_message_id: str, progress_text: str) -> None:
        """Append progress logs and update process card for one inbound message."""
        if source_message_id not in self._processing_cards:
            return
        normalized = (progress_text or "").strip()
        if not normalized:
            return
        logs = self._processing_card_logs.setdefault(source_message_id, [])
        if not logs or logs[-1] != normalized:
            logs.append(normalized)
        self._processing_card_step[source_message_id] = int(
            self._processing_card_step.get(source_message_id, 0)
        ) + 1
        await self._render_processing_card(source_message_id)

    async def _delete_processing_card_for_message(self, source_message_id: str) -> None:
        """Delete and clear process card for one inbound message."""
        card_message_id = self._processing_cards.get(source_message_id)
        if not card_message_id:
            return
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._delete_message_sync, card_message_id)
        finally:
            self._processing_cards.pop(source_message_id, None)
            self._processing_card_text.pop(source_message_id, None)
            self._processing_card_logs.pop(source_message_id, None)
            self._processing_card_step.pop(source_message_id, None)
            animator = self._processing_card_animators.pop(source_message_id, None)
            if animator:
                animator.cancel()

    # Max character length for a single message in AI context (history / parent quote)
    _CONTEXT_MAX_TEXT_LEN = 300

    # Strip Feishu @mention XML tags from raw text messages
    _AT_TAG_RE = re.compile(r"<at[^>]*>.*?</at>", re.DOTALL)

    # Regex to match markdown tables (header + separator + data rows)
    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    _CODE_BLOCK_RE = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown table into a Feishu table element."""
        lines = [_line.strip() for _line in table_text.strip().split("\n") if _line.strip()]
        if len(lines) < 3:
            return None
        def split(_line: str) -> list[str]:
            return [c.strip() for c in _line.strip("|").split("|")]
        headers = split(lines[0])
        rows = [split(_line) for _line in lines[2:]]
        columns = [{"tag": "column", "name": f"c{i}", "display_name": h, "width": "auto"}
                   for i, h in enumerate(headers)]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [{f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))} for r in rows],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Split content into div/markdown + table elements for Feishu card."""
        elements, last_end = [], 0
        for m in self._TABLE_RE.finditer(content):
            before = content[last_end:m.start()]
            if before.strip():
                elements.extend(self._split_headings(before))
            elements.append(self._parse_md_table(m.group(1)) or {"tag": "markdown", "content": m.group(1)})
            last_end = m.end()
        remaining = content[last_end:]
        if remaining.strip():
            elements.extend(self._split_headings(remaining))
        return elements or [{"tag": "markdown", "content": content}]

    def _split_headings(self, content: str) -> list[dict]:
        """Split content by headings, converting headings to div elements."""
        protected = content
        code_blocks = []
        for m in self._CODE_BLOCK_RE.finditer(content):
            code_blocks.append(m.group(1))
            protected = protected.replace(m.group(1), f"\x00CODE{len(code_blocks)-1}\x00", 1)

        elements = []
        last_end = 0
        for m in self._HEADING_RE.finditer(protected):
            before = protected[last_end:m.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            text = m.group(2).strip()
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**{text}**",
                },
            })
            last_end = m.end()
        remaining = protected[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})

        for i, cb in enumerate(code_blocks):
            for el in elements:
                if el.get("tag") == "markdown":
                    el["content"] = el["content"].replace(f"\x00CODE{i}\x00", cb)

        return elements or [{"tag": "markdown", "content": content}]

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".tif"}
    _AUDIO_EXTS = {".opus"}
    _FILE_TYPE_MAP = {
        ".opus": "opus", ".mp4": "mp4", ".pdf": "pdf", ".doc": "doc", ".docx": "doc",
        ".xls": "xls", ".xlsx": "xls", ".ppt": "ppt", ".pptx": "ppt",
    }

    def _upload_image_sync(self, file_path: str) -> str | None:
        """Upload an image to Feishu and return the image_key."""
        try:
            with open(file_path, "rb") as f:
                request = CreateImageRequest.builder() \
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(f)
                        .build()
                    ).build()
                response = self._client.im.v1.image.create(request)
                if response.success():
                    image_key = response.data.image_key
                    logger.debug("Uploaded image {}: {}", os.path.basename(file_path), image_key)
                    return image_key
                else:
                    logger.error("Failed to upload image: code={}, msg={}", response.code, response.msg)
                    return None
        except Exception as e:
            logger.error("Error uploading image {}: {}", file_path, e)
            return None

    @staticmethod
    def _probe_duration_ms(file_path: str) -> int | None:
        """Try to detect media duration with ffprobe; return None when unavailable."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    file_path,
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None
        try:
            seconds = float((result.stdout or "").strip())
        except Exception:
            return None
        ms = int(seconds * 1000)
        return ms if ms > 0 else None

    def _upload_file_sync(self, file_path: str, duration_ms: int | None = None) -> str | None:
        """Upload a file to Feishu and return the file_key."""
        ext = os.path.splitext(file_path)[1].lower()
        file_type = self._FILE_TYPE_MAP.get(ext, "stream")
        file_name = os.path.basename(file_path)
        try:
            file_size = os.path.getsize(file_path)
        except Exception as e:
            logger.error("Error reading file size {}: {}", file_path, e)
            return None
        if file_size <= 0:
            logger.error("Cannot upload empty file: {}", file_path)
            return None
        if file_size > self._MAX_UPLOAD_FILE_BYTES:
            logger.error(
                "File too large for Feishu upload (max 30MB): {} ({} bytes)",
                file_path,
                file_size,
            )
            return None
        try:
            with open(file_path, "rb") as f:
                builder = (
                    CreateFileRequestBody.builder()
                    .file_type(file_type)
                    .file_name(file_name)
                    .file(f)
                )
                if duration_ms and duration_ms > 0 and file_type in {"opus", "mp4"}:
                    builder = builder.duration(duration_ms)
                request = CreateFileRequest.builder().request_body(builder.build()).build()
                response = self._client.im.v1.file.create(request)
                if response.success():
                    file_key = response.data.file_key
                    logger.debug("Uploaded file {}: {}", file_name, file_key)
                    return file_key
                else:
                    logger.error("Failed to upload file: code={}, msg={}", response.code, response.msg)
                    return None
        except Exception as e:
            logger.error("Error uploading file {}: {}", file_path, e)
            return None

    def _download_image_sync(self, message_id: str, image_key: str) -> tuple[bytes | None, str | None]:
        """Download an image from Feishu message by message_id and image_key."""
        try:
            request = GetMessageResourceRequest.builder() \
                .message_id(message_id) \
                .file_key(image_key) \
                .type("image") \
                .build()
            response = self._client.im.v1.message_resource.get(request)
            if response.success():
                file_data = response.file
                # GetMessageResourceRequest returns BytesIO, need to read bytes
                if hasattr(file_data, 'read'):
                    file_data = file_data.read()
                return file_data, response.file_name
            else:
                logger.error("Failed to download image: code={}, msg={}", response.code, response.msg)
                return None, None
        except Exception as e:
            logger.error("Error downloading image {}: {}", image_key, e)
            return None, None

    def _download_file_sync(
        self, message_id: str, file_key: str, resource_type: str = "file"
    ) -> tuple[bytes | None, str | None]:
        """Download a file/audio/media from a Feishu message by message_id and file_key."""
        try:
            request = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type(resource_type)
                .build()
            )
            response = self._client.im.v1.message_resource.get(request)
            if response.success():
                file_data = response.file
                if hasattr(file_data, "read"):
                    file_data = file_data.read()
                return file_data, response.file_name
            else:
                logger.error("Failed to download {}: code={}, msg={}", resource_type, response.code, response.msg)
                return None, None
        except Exception:
            logger.exception("Error downloading {} {}", resource_type, file_key)
            return None, None

    async def _download_and_save_media(
        self,
        msg_type: str,
        content_json: dict,
        message_id: str | None = None
    ) -> tuple[str | None, str]:
        """
        Download media from Feishu and save to local disk.

        Returns:
            (file_path, content_text) - file_path is None if download failed
        """
        loop = asyncio.get_running_loop()
        media_dir = Path.home() / ".nanobot" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        data, filename = None, None

        if msg_type == "image":
            image_key = content_json.get("image_key")
            if image_key and message_id:
                data, filename = await loop.run_in_executor(
                    None, self._download_image_sync, message_id, image_key
                )
                if not filename:
                    filename = f"{image_key[:16]}.jpg"

        elif msg_type in ("audio", "file", "media"):
            file_key = content_json.get("file_key")
            if file_key and message_id:
                data, filename = await loop.run_in_executor(
                    None, self._download_file_sync, message_id, file_key, msg_type
                )
                if not filename:
                    ext = {"audio": ".opus", "media": ".mp4"}.get(msg_type, "")
                    filename = f"{file_key[:16]}{ext}"

        if data and filename:
            file_path = media_dir / filename
            file_path.write_bytes(data)
            logger.debug("Downloaded {} to {}", msg_type, file_path)
            return str(file_path), f"[{msg_type}: {filename}]"

        return None, f"[{msg_type}: download failed]"

    def _send_message_sync(self, receive_id_type: str, receive_id: str, msg_type: str, content: str) -> str | None:
        """Send a single message (text/image/file/interactive) synchronously and return message_id."""
        try:
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type(msg_type)
                    .content(content)
                    .build()
                ).build()
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(
                    "Failed to send Feishu {} message: code={}, msg={}, log_id={}",
                    msg_type, response.code, response.msg, response.get_log_id()
                )
                return None
            logger.debug("Feishu {} message sent to {}", msg_type, receive_id)
            return getattr(response.data, "message_id", None)
        except Exception as e:
            logger.error("Error sending Feishu {} message: {}", msg_type, e)
            return None

    def _reply_message_sync(self, parent_message_id: str, msg_type: str, content: str) -> str | None:
        """Reply to a Feishu message natively (thread reply) and return message_id."""
        try:
            from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody
        except ImportError:
            logger.warning("ReplyMessageRequest not available; falling back to normal send")
            return None

        try:
            request = ReplyMessageRequest.builder() \
                .message_id(parent_message_id) \
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type(msg_type)
                    .content(content)
                    .build()
                ).build()
            response = self._client.im.v1.message.reply(request)
            if not response.success():
                logger.error(
                    "Failed to reply Feishu {} message: code={}, msg={}, log_id={}",
                    msg_type, response.code, response.msg, response.get_log_id()
                )
                return None
            logger.debug("Feishu {} message replied to {}", msg_type, parent_message_id)
            return getattr(response.data, "message_id", None)
        except Exception as e:
            logger.error("Error replying Feishu {} message: {}", msg_type, e)
            return None

    def _send_with_optional_reply_sync(
        self,
        receive_id_type: str,
        receive_id: str,
        msg_type: str,
        content: str,
        reply_to_message_id: str | None = None,
    ) -> str | None:
        """Reply in-thread when possible; always fallback to normal send for reliability."""
        if reply_to_message_id:
            replied_message_id = self._reply_message_sync(reply_to_message_id, msg_type, content)
            if replied_message_id:
                return replied_message_id
            logger.warning(
                "Feishu reply failed, fallback to normal send: parent_message_id={} msg_type={}",
                reply_to_message_id,
                msg_type,
            )
        return self._send_message_sync(receive_id_type, receive_id, msg_type, content)

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu, including media (images/files) if present."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return

        try:
            receive_id_type = "chat_id" if msg.chat_id.startswith("oc_") else "open_id"
            receive_id = msg.chat_id
            loop = asyncio.get_running_loop()
            source_message_id = str(msg.metadata.get("message_id", "")).strip()
            is_turn_done = bool(msg.metadata.get("_turn_done"))
            is_progress = bool(msg.metadata.get("_progress"))
            force_msg_type = str(msg.metadata.get("feishu_msg_type", "")).strip().lower()
            disable_reply_quote = bool(msg.metadata.get("feishu_disable_reply_quote")) or force_msg_type == "sticker"
            reply_to_message_id = source_message_id if (
                self.config.reply_to_message and source_message_id and not is_progress and not disable_reply_quote
            ) else None
            is_tool_hint = bool(msg.metadata.get("_tool_hint"))
            force_content = msg.metadata.get("feishu_content")
            force_system_content = msg.metadata.get("feishu_system_content")

            if is_progress:
                if source_message_id:
                    progress_prefix = "正在调用工具： " if is_tool_hint else "处理中： "
                    await self._update_processing_card_for_message(source_message_id, f"{progress_prefix}{msg.content}")
                return

            for file_path in msg.media:
                if not os.path.isfile(file_path):
                    logger.warning("Media file not found: {}", file_path)
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self._IMAGE_EXTS:
                    key = await loop.run_in_executor(None, self._upload_image_sync, file_path)
                    if key:
                        await loop.run_in_executor(
                            None, self._send_with_optional_reply_sync,
                            receive_id_type, receive_id, "image",
                            json.dumps({"image_key": key}, ensure_ascii=False),
                            reply_to_message_id,
                        )
                else:
                    duration_ms = self._probe_duration_ms(file_path) if ext in {".opus", ".mp4"} else None
                    if ext in self._NON_OPUS_AUDIO_EXTS:
                        logger.warning(
                            "Non-OPUS audio '{}' will be sent as file. Convert to .opus to send as audio.",
                            os.path.basename(file_path),
                        )
                    key = await loop.run_in_executor(None, self._upload_file_sync, file_path, duration_ms)
                    if key:
                        media_type = "audio" if ext in self._AUDIO_EXTS else "file"
                        await loop.run_in_executor(
                            None, self._send_with_optional_reply_sync,
                            receive_id_type, receive_id, media_type,
                            json.dumps({"file_key": key}, ensure_ascii=False),
                            reply_to_message_id,
                        )

            if force_msg_type:
                payload = force_content
                if payload is None and force_msg_type == "system":
                    payload = force_system_content
                if payload is None and force_msg_type == "text":
                    payload = {"text": msg.content}

                if payload is None:
                    if msg.content and msg.content.strip():
                        logger.warning(
                            "feishu_msg_type={} provided without feishu_content; fallback to plain text",
                            force_msg_type,
                        )
                        await loop.run_in_executor(
                            None,
                            self._send_with_optional_reply_sync,
                            receive_id_type,
                            receive_id,
                            "text",
                            json.dumps({"text": msg.content}, ensure_ascii=False),
                            reply_to_message_id,
                        )
                    else:
                        logger.warning(
                            "feishu_msg_type={} provided without feishu_content and no fallback text",
                            force_msg_type,
                        )
                else:
                    if not isinstance(payload, str):
                        payload = json.dumps(payload, ensure_ascii=False)
                    await loop.run_in_executor(
                        None,
                        self._send_with_optional_reply_sync,
                        receive_id_type,
                        receive_id,
                        force_msg_type,
                        payload,
                        reply_to_message_id,
                    )
            elif msg.content and msg.content.strip():
                await loop.run_in_executor(
                    None, self._send_with_optional_reply_sync,
                    receive_id_type, receive_id, "text",
                    json.dumps({"text": msg.content}, ensure_ascii=False),
                    reply_to_message_id,
                )

            # Only clear reaction when agent explicitly marks this turn as complete.
            if source_message_id and is_turn_done:
                await self._delete_processing_card_for_message(source_message_id)
                await self._delete_reaction_for_message(source_message_id)

        except Exception as e:
            logger.error("Error sending Feishu message: {}", e)

    @classmethod
    def _extract_text_at_user_ids(cls, text: str) -> set[str]:
        if not text:
            return set()
        return {m.group(1).strip() for m in cls._AT_USER_ID_RE.finditer(text) if m.group(1).strip()}

    @classmethod
    def _extract_json_at_user_ids(cls, obj: Any) -> set[str]:
        ids: set[str] = set()
        if isinstance(obj, dict):
            tag = str(obj.get("tag", "")).lower()
            if tag == "at":
                uid = str(obj.get("user_id", "")).strip()
                if uid:
                    ids.add(uid)
            for value in obj.values():
                ids.update(cls._extract_json_at_user_ids(value))
        elif isinstance(obj, list):
            for item in obj:
                ids.update(cls._extract_json_at_user_ids(item))
        return ids

    @staticmethod
    def _extract_sdk_mention_ids(message: Any) -> set[str]:
        ids: set[str] = set()
        mentions = getattr(message, "mentions", None)
        if not isinstance(mentions, list):
            return ids

        for mention in mentions:
            if mention is None:
                continue
            # Common SDK shapes: mention.id.open_id / mention.id.user_id / mention.open_id / mention.user_id
            mention_id = getattr(mention, "id", None)
            for raw in (
                getattr(mention_id, "open_id", None),
                getattr(mention_id, "user_id", None),
                getattr(mention_id, "union_id", None),
                getattr(mention, "open_id", None),
                getattr(mention, "user_id", None),
                getattr(mention, "union_id", None),
                getattr(mention, "key", None),
            ):
                value = str(raw or "").strip()
                if value:
                    ids.add(value)
        return ids

    def _extract_mentioned_user_ids(self, message: Any, msg_type: str, content_json: dict) -> set[str]:
        ids = self._extract_sdk_mention_ids(message)
        if msg_type == "text":
            ids.update(self._extract_text_at_user_ids(str(content_json.get("text", ""))))
        elif msg_type == "post":
            ids.update(self._extract_json_at_user_ids(content_json))
        return ids

    def _is_group_allowed(self, chat_id: str) -> bool:
        policy = str(self.config.group_policy or "mention").lower()
        if policy == "allowlist":
            return chat_id in (self.config.group_allow_from or [])
        return True

    def _should_respond_in_group(
        self,
        *,
        chat_id: str,
        sender_id: str,
        message: Any,
        msg_type: str,
        content_json: dict,
    ) -> tuple[bool, bool, bool]:
        """Return (should_reply, was_mentioned, proactive_hit)."""
        policy = str(self.config.group_policy or "mention").lower()

        if policy == "allowlist":
            return self._is_group_allowed(chat_id), False, False
        if policy == "open":
            return True, False, False

        mentioned_ids = self._extract_mentioned_user_ids(message, msg_type, content_json)
        bot_open_id = str(self._bot_open_id or self.config.bot_open_id or "").strip()
        if not bot_open_id:
            candidates = {uid for uid in mentioned_ids if uid not in {"all", sender_id}}
            if len(candidates) == 1:
                bot_open_id = next(iter(candidates))
                self._bot_open_id = bot_open_id
                self.config.bot_open_id = bot_open_id
        if bot_open_id:
            was_mentioned = bot_open_id in mentioned_ids
        else:
            # Fallback: when bot_open_id is unknown, treat any explicit @ as mention trigger.
            was_mentioned = any(uid and uid != "all" for uid in mentioned_ids)

        if not was_mentioned and self.config.allow_room_mentions and "all" in mentioned_ids:
            was_mentioned = True

        if was_mentioned:
            return True, True, False

        try:
            p = float(self.config.proactive_reply_probability or 0.0)
        except Exception:
            p = 0.0
        p = max(0.0, min(1.0, p))
        proactive_hit = p > 0.0 and random.random() < p
        return proactive_hit, False, proactive_hit

    @classmethod
    def _clean_at_tags(cls, text: str) -> str:
        """Remove Feishu @mention XML tags from raw text, returning clean content."""
        return cls._AT_TAG_RE.sub("", text).strip()

    def _extract_message_text(self, msg_type: str, content_json: dict) -> str:
        """Extract plain text from a message content dict (used for context fetching).

        Cleans @mention tags and truncates to _CONTEXT_MAX_TEXT_LEN.
        """
        if msg_type == "text":
            text = self._clean_at_tags(str(content_json.get("text", "")).strip())
        elif msg_type == "post":
            raw, _ = _extract_post_content(content_json)
            text = raw or ""
        elif msg_type in ("image", "audio", "file", "media", "sticker"):
            text = MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")
        elif msg_type in ("share_chat", "share_user", "interactive", "share_calendar_event", "system", "merge_forward"):
            text = _extract_share_card_content(content_json, msg_type)
        else:
            text = f"[{msg_type}]"
        if len(text) > self._CONTEXT_MAX_TEXT_LEN:
            text = text[:self._CONTEXT_MAX_TEXT_LEN] + "..."
        return text

    @staticmethod
    def _extract_sender_open_id(sender: Any) -> str:
        """Extract sender open_id robustly across SDK payload shapes."""
        if not sender:
            return ""
        direct_candidates = (
            getattr(sender, "id", None),
            getattr(sender, "open_id", None),
            getattr(sender, "user_id", None),
        )
        for candidate in direct_candidates:
            value = str(candidate or "").strip()
            if value:
                return value
        sender_id = getattr(sender, "sender_id", None)
        nested_candidates = (
            getattr(sender_id, "open_id", None),
            getattr(sender_id, "user_id", None),
            getattr(sender_id, "union_id", None),
        )
        for candidate in nested_candidates:
            value = str(candidate or "").strip()
            if value:
                return value
        return ""

    def _fetch_group_context_sync(
        self, chat_id: str, count: int, current_msg_id: str
    ) -> list[tuple[str, str]]:
        """Fetch recent group messages for AI context. Returns [(open_id, text), ...] in chronological order."""
        try:
            from lark_oapi.api.im.v1 import ListMessageRequest
        except ImportError:
            logger.warning("lark_oapi ListMessageRequest not available; group context disabled")
            return []

        try:
            # Build request — use only the universally supported builder methods.
            # sort_type / page_size are optional; skip them if the builder rejects them.
            builder = (
                ListMessageRequest.builder()
                .container_id_type("chat")
                .container_id(chat_id)
            )
            try:
                builder = builder.sort_type("ByCreateTimeDesc")
            except Exception:
                pass  # older SDK versions may not support sort_type
            try:
                builder = builder.page_size(min(count + 5, 50))
            except Exception:
                pass
            request = builder.build()

            logger.info("Feishu: fetching group context for chat={} count={}", chat_id, count)
            response = self._client.im.v1.message.list(request)
            if not response.success():
                logger.error(
                    "Feishu: failed to fetch group context: code={} msg={} detail={}",
                    response.code,
                    response.msg,
                    getattr(response, "error", ""),
                )
                return []

            items = list(getattr(response.data, "items", None) or [])
            logger.info("Feishu: group context API returned {} items", len(items))
            result: list[tuple[str, str]] = []
            for item in items:
                if getattr(item, "message_id", None) == current_msg_id:
                    continue  # exclude current triggering message
                sender = getattr(item, "sender", None)
                sender_type = str(getattr(sender, "sender_type", "user") or "")
                if sender_type == "bot":
                    continue  # exclude bot replies

                sender_id = self._extract_sender_open_id(sender)

                msg_type = str(getattr(item, "msg_type", "text") or "text")
                body = getattr(item, "body", None)
                body_content = str(getattr(body, "content", "") or "") if body else ""
                try:
                    content_json = json.loads(body_content) if body_content else {}
                except json.JSONDecodeError:
                    content_json = {}

                text = self._extract_message_text(msg_type, content_json)
                if text:
                    result.append((sender_id, text))
                if len(result) >= count:
                    break

            result.reverse()  # oldest first → chronological order
            logger.info("Feishu: group context assembled {} usable messages", len(result))
            return result
        except Exception:
            logger.exception("Feishu: exception in _fetch_group_context_sync for chat={}", chat_id)
            return []

    async def _fetch_group_context(
        self, chat_id: str, count: int, current_msg_id: str
    ) -> list[tuple[str, str]]:
        """Async wrapper around _fetch_group_context_sync."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._fetch_group_context_sync, chat_id, count, current_msg_id
        )

    def _fetch_parent_message_sync(self, parent_id: str) -> tuple[str, str] | None:
        """Fetch the quoted/replied-to message. Returns (open_id, text) or None."""
        try:
            from lark_oapi.api.im.v1 import GetMessageRequest
        except ImportError:
            return None
        try:
            request = GetMessageRequest.builder().message_id(parent_id).build()
            response = self._client.im.v1.message.get(request)
            if not response.success():
                logger.debug("Failed to fetch parent message {}: {}", parent_id, response.msg)
                return None
            items = list(getattr(response.data, "items", None) or [])
            if not items:
                return None
            item = items[0]
            sender = getattr(item, "sender", None)
            open_id = self._extract_sender_open_id(sender)
            msg_type = str(getattr(item, "msg_type", "text") or "text")
            body = getattr(item, "body", None)
            body_content = str(getattr(body, "content", "") or "") if body else ""
            try:
                content_json = json.loads(body_content) if body_content else {}
            except json.JSONDecodeError:
                content_json = {}
            text = self._extract_message_text(msg_type, content_json)
            return (open_id, text) if text else None
        except Exception as e:
            logger.debug("Error fetching parent message {}: {}", parent_id, e)
            return None

    async def _fetch_parent_message(self, parent_id: str) -> tuple[str, str] | None:
        """Async wrapper around _fetch_parent_message_sync."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_parent_message_sync, parent_id)

    def _fetch_user_name_sync(self, open_id: str) -> str | None:
        """Fetch the display name for a user open_id via Feishu contact API."""
        try:
            from lark_oapi.api.contact.v3 import GetUserRequest
        except ImportError:
            return None
        try:
            request = (
                GetUserRequest.builder()
                .user_id(open_id)
                .user_id_type("open_id")
                .build()
            )
            response = self._client.contact.v3.user.get(request)
            if not response.success():
                return None
            user = getattr(response.data, "user", None)
            name = str(getattr(user, "name", "") or "").strip() if user else ""
            return name or None
        except Exception as e:
            logger.debug("Failed to fetch display name for {}: {}", open_id, e)
            return None

    async def _get_user_display_name(self, open_id: str) -> str:
        """Return cached display name for open_id, fetching from API on first access.

        Falls back to a short suffix of the open_id if the API call fails or
        the contact permission is not granted.
        """
        if open_id in self._user_name_cache:
            return self._user_name_cache[open_id]
        short = open_id[-6:] if len(open_id) > 6 else (open_id or "?")
        loop = asyncio.get_running_loop()
        name = await loop.run_in_executor(None, self._fetch_user_name_sync, open_id)
        display = name if name else f"用户_{short}"
        self._user_name_cache[open_id] = display
        # Keep cache bounded
        while len(self._user_name_cache) > 500:
            self._user_name_cache.pop(next(iter(self._user_name_cache)))
        return display

    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
        """
        Sync handler for incoming messages (called from WebSocket thread).
        Schedules async handling in the main event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    async def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        """Handle incoming message from Feishu."""
        try:
            event = data.event
            message = event.message
            sender = event.sender

            # Deduplication check
            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None

            # Trim cache
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

            # Skip bot messages
            if sender.sender_type == "bot":
                return

            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type
            msg_type = message.message_type
            is_group = chat_type == "group"
            reply_to = chat_id if is_group else sender_id
            receive_id_type = "chat_id" if reply_to.startswith("oc_") else "open_id"

            try:
                content_json = json.loads(message.content) if message.content else {}
            except json.JSONDecodeError:
                content_json = {}
            text_content = str(content_json.get("text", "")).strip() if msg_type == "text" else ""
            is_slash_command = bool(text_content.startswith("/"))
            was_mentioned = False
            proactive_reply = False

            if is_group:
                if is_slash_command:
                    if not self._is_group_allowed(chat_id):
                        return
                else:
                    should_reply, was_mentioned, proactive_reply = self._should_respond_in_group(
                        chat_id=chat_id,
                        sender_id=sender_id,
                        message=message,
                        msg_type=msg_type,
                        content_json=content_json,
                    )
                    if not should_reply:
                        return

            # Permission check before creating any visible artifacts (reaction / progress card).
            if not self.is_allowed(sender_id):
                logger.warning("Feishu: access denied for sender {}", sender_id)
                return

            # Add reaction and track it for completion cleanup.
            reaction_id = await self._add_reaction(message_id, self.config.react_emoji)
            if reaction_id:
                self._reaction_ids[message_id] = reaction_id
                # Bound reaction cache size to avoid unbounded growth.
                while len(self._reaction_ids) > 1000:
                    self._reaction_ids.popitem(last=False)
            if not is_slash_command:
                await self._create_processing_card_for_message(message_id, receive_id_type, reply_to)

            # Parse content
            content_parts = []
            media_paths = []

            if msg_type == "text":
                text = content_json.get("text", "")
                if text:
                    content_parts.append(self._clean_at_tags(text))

            elif msg_type == "post":
                text, image_keys = _extract_post_content(content_json)
                if text:
                    content_parts.append(text)
                # Download images embedded in post
                for img_key in image_keys:
                    file_path, content_text = await self._download_and_save_media(
                        "image", {"image_key": img_key}, message_id
                    )
                    if file_path:
                        media_paths.append(file_path)
                    content_parts.append(content_text)

            elif msg_type in ("image", "audio", "file", "media"):
                file_path, content_text = await self._download_and_save_media(msg_type, content_json, message_id)
                if file_path:
                    media_paths.append(file_path)
                content_parts.append(content_text)

            elif msg_type in ("share_chat", "share_user", "interactive", "share_calendar_event", "system", "merge_forward"):
                # Handle share cards and interactive messages
                text = _extract_share_card_content(content_json, msg_type)
                if text:
                    content_parts.append(text)

            else:
                content_parts.append(MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]"))

            content = "\n".join(content_parts) if content_parts else ""

            if not content and not media_paths:
                return

            # Inject sender label so the AI knows who is speaking in group chats (问题4+5)
            if is_group and content:
                sender_name = await self._get_user_display_name(sender_id)
                content = f"[{sender_name}]: {content}"

            # Build context prefix: recent group history + quoted/replied-to message
            context_prefix_parts: list[str] = []

            if is_group and self.config.group_context_count > 0:
                context_msgs = await self._fetch_group_context(
                    chat_id, self.config.group_context_count, message_id
                )
                if context_msgs:
                    unique_ids = list({oid for oid, _ in context_msgs})
                    names = await asyncio.gather(
                        *[self._get_user_display_name(oid) for oid in unique_ids]
                    )
                    name_map = dict(zip(unique_ids, names))
                    history_lines = "\n".join(
                        f"[{name_map.get(oid, oid[-6:])}]: {text}" for oid, text in context_msgs
                    )
                    context_prefix_parts.append(f"[群聊近期消息]\n{history_lines}")

            # If this message is a reply, include the quoted message (问题3)
            parent_id = str(getattr(message, "parent_id", "") or "").strip()
            if parent_id:
                parent_msg = await self._fetch_parent_message(parent_id)
                if parent_msg:
                    p_oid, p_text = parent_msg
                    p_name = await self._get_user_display_name(p_oid)
                    context_prefix_parts.append(f"[被引用的消息 - {p_name}]\n{p_text}")

            if context_prefix_parts:
                content = "\n\n".join(context_prefix_parts) + f"\n\n[当前消息]\n{content}"

            # Forward to message bus
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                media=media_paths,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                    "is_group": is_group,
                    "group_sender_id": sender_id if is_group else "",
                    "was_mentioned": was_mentioned,
                    "proactive_reply": proactive_reply,
                    "bot_open_id": self._bot_open_id or "",
                }
            )

        except Exception as e:
            logger.error("Error processing Feishu message: {}", e)
