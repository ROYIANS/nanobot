---
name: feishu-sticker
description: Send Feishu sticker messages using message tool with sticker_file_key.
---

# Feishu Sticker

Use this skill when the user asks to send a Feishu sticker message.

## How to send

Call `message` with:
- `channel="feishu"`
- `chat_id` from current context (or explicit target)
- `sticker_file_key` set to the sticker `file_key`
- `content` can be empty string

Example:

```json
{
  "content": "",
  "channel": "feishu",
  "chat_id": "oc_xxx",
  "sticker_file_key": "file_v2_xxx"
}
```

Reuse latest saved sticker in current chat:

```json
{
  "content": "",
  "channel": "feishu",
  "chat_id": "oc_xxx",
  "use_recent_sticker": true
}
```

## Notes

- `sticker_file_key` auto-sets `msg_type="sticker"` and payload `{"file_key":"..."}`.
- Sticker sending is forced to **non-quote mode** (no reply reference to source message).
- Received sticker keys are persisted at `~/.nanobot/feishu_stickers.json` by chat.
- If user asks to reuse "just sent sticker", read latest key from that store for current chat.
