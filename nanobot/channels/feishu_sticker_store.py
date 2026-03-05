"""Persistent store for Feishu sticker file keys."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STORE_VERSION = 1
MAX_PER_CHAT = 100


def default_store_path() -> Path:
    return Path.home() / ".nanobot" / "feishu_stickers.json"


def load_store(path: Path | None = None) -> dict[str, Any]:
    p = path or default_store_path()
    if not p.exists():
        return {"version": STORE_VERSION, "chats": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"version": STORE_VERSION, "chats": {}}
    if not isinstance(data, dict):
        return {"version": STORE_VERSION, "chats": {}}
    chats = data.get("chats")
    if not isinstance(chats, dict):
        chats = {}
    return {"version": STORE_VERSION, "chats": chats}


def save_store(store: dict[str, Any], path: Path | None = None) -> None:
    p = path or default_store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def record_sticker(
    *,
    chat_id: str,
    sender_id: str,
    file_key: str,
    message_id: str = "",
    create_time_ms: str = "",
    path: Path | None = None,
) -> None:
    file_key = str(file_key or "").strip()
    chat_id = str(chat_id or "").strip()
    if not file_key or not chat_id:
        return

    store = load_store(path)
    chats = store.setdefault("chats", {})
    entries = list(chats.get(chat_id) or [])

    # Deduplicate by file_key and keep newest at end.
    entries = [e for e in entries if str((e or {}).get("file_key") or "").strip() != file_key]
    entries.append(
        {
            "file_key": file_key,
            "sender_id": str(sender_id or "").strip(),
            "message_id": str(message_id or "").strip(),
            "create_time": str(create_time_ms or "").strip(),
        }
    )
    if len(entries) > MAX_PER_CHAT:
        entries = entries[-MAX_PER_CHAT:]

    chats[chat_id] = entries
    save_store(store, path)


def list_chat_stickers(chat_id: str, *, limit: int = 20, path: Path | None = None) -> list[dict[str, str]]:
    chat_id = str(chat_id or "").strip()
    if not chat_id:
        return []
    store = load_store(path)
    chats = store.get("chats") or {}
    items = list(chats.get(chat_id) or [])
    if limit > 0:
        items = items[-limit:]
    return list(reversed(items))


def latest_chat_sticker(chat_id: str, *, path: Path | None = None) -> dict[str, str] | None:
    items = list_chat_stickers(chat_id, limit=1, path=path)
    return items[0] if items else None

