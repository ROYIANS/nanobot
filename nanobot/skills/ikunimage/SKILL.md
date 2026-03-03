---
name: ikunimage
description: Generate or edit images via ikuncode (Gemini image API).
---

# Ikunimage

Use this when the user asks to:
- generate an image
- edit an existing image
- produce poster/visual/media assets

## Tool

Use the `ikun_image` tool.

### Text-to-image
- Required: `prompt`
- Optional: `aspect_ratio`, `size`, `output_path`, `retry`

### Image edit (img2img)
- Required: `prompt`, `input_image`
- Optional: `aspect_ratio`, `output_path`, `retry`

## Delivery

After `ikun_image` returns a local path:
- If the user is on a chat channel and expects direct delivery, send the file with `message` tool using `media`.
- Otherwise return the saved path and ask whether to send it out.
