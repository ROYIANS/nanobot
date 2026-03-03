"""Ikun image generation tool backed by ikuncode NanoBananaPro API."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool


class IkunImageTool(Tool):
    """Generate or edit images using ikuncode's Gemini image endpoint."""

    _VALID_ASPECT_RATIOS = (
        "1:1",
        "16:9",
        "9:16",
        "4:3",
        "3:4",
        "3:2",
        "2:3",
        "21:9",
        "5:4",
        "4:5",
    )
    _VALID_SIZES = ("1K", "2K", "4K")
    _TIMEOUT_BY_SIZE = {"1K": 360, "2K": 600, "4K": 1200}
    _MAX_EDIT_INPUT_BYTES = 4 * 1024 * 1024
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        workspace: Path,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str = "gemini-3-pro-image-preview",
    ) -> None:
        self.workspace = workspace
        self.api_key = (api_key or "").strip()
        self.api_base = (api_base or "https://api.ikuncode.cc").rstrip("/")
        self.model = (model or "gemini-3-pro-image-preview").strip()

    @property
    def name(self) -> str:
        return "ikun_image"

    @property
    def description(self) -> str:
        return (
            "Generate or edit an image with ikuncode (Gemini image). "
            "Returns the saved local file path."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 1},
                "input_image": {
                    "type": "string",
                    "description": "Optional local image path for image editing (img2img).",
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": list(self._VALID_ASPECT_RATIOS),
                    "description": "Output aspect ratio.",
                },
                "size": {
                    "type": "string",
                    "enum": list(self._VALID_SIZES),
                    "description": "Output quality for text-to-image only.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional output path; relative paths are under workspace.",
                },
                "retry": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Max retry times for transient failures.",
                },
                "api_key": {
                    "type": "string",
                    "description": "Optional API key override for this call.",
                },
            },
            "required": ["prompt"],
        }

    async def execute(
        self,
        prompt: str,
        input_image: str | None = None,
        aspect_ratio: str = "1:1",
        size: str = "2K",
        output_path: str | None = None,
        retry: int = 3,
        api_key: str | None = None,
        **_: Any,
    ) -> str:
        key = self._resolve_api_key(api_key)
        if not key:
            return (
                "Error: ikuncode API key not configured. "
                "Set providers.ikuncode.api_key, IKUN_API_KEY, IKUNCODE_API_KEY, "
                "or ~/.ikunimage/config.json."
            )

        prompt = prompt.strip()
        if not prompt:
            return "Error: prompt cannot be empty"

        aspect_ratio = (aspect_ratio or "1:1").strip()
        if aspect_ratio not in self._VALID_ASPECT_RATIOS:
            return (
                "Error: invalid aspect_ratio. Supported: "
                + ", ".join(self._VALID_ASPECT_RATIOS)
            )

        size = (size or "2K").strip().upper()
        if size not in self._VALID_SIZES:
            return "Error: invalid size. Supported: 1K, 2K, 4K"

        payload, err = self._build_payload(
            prompt=prompt,
            input_image=input_image,
            aspect_ratio=aspect_ratio,
            size=size,
        )
        if err:
            return err
        assert payload is not None

        timeout = self._TIMEOUT_BY_SIZE.get(size, 600)
        data, err = await self._request_with_retry(
            payload=payload,
            api_key=key,
            timeout_s=timeout,
            max_retries=max(0, retry),
        )
        if err:
            return err
        assert data is not None

        image_bytes, mime_type, err = self._extract_image(data)
        if err:
            return err

        destination = self._resolve_output_path(output_path, prompt, mime_type)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)

        size_kb = len(image_bytes) / 1024
        return f"Image generated: {destination.resolve()} ({size_kb:.1f} KB)"

    def _resolve_api_key(self, override: str | None) -> str:
        if override and override.strip():
            return override.strip()
        if self.api_key:
            return self.api_key

        for env in ("IKUN_API_KEY", "IKUNCODE_API_KEY"):
            value = os.environ.get(env, "").strip()
            if value:
                return value

        config_file = Path.home() / ".ikunimage" / "config.json"
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
            except Exception:
                return ""
            key = str((data or {}).get("api_key") or "").strip()
            if key:
                return key
        return ""

    def _build_payload(
        self,
        prompt: str,
        input_image: str | None,
        aspect_ratio: str,
        size: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if input_image:
            image_path = Path(input_image)
            if not image_path.is_absolute():
                image_path = self.workspace / image_path
            if not image_path.exists() or not image_path.is_file():
                return None, f"Error: input_image not found: {input_image}"
            if image_path.stat().st_size > self._MAX_EDIT_INPUT_BYTES:
                return None, "Error: input_image is too large (> 4MB)"
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/jpeg"
            encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            return {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": mime_type, "data": encoded}},
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"aspectRatio": aspect_ratio},
                },
            }, None

        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": aspect_ratio, "image_size": size},
            },
        }, None

    async def _request_with_retry(
        self,
        payload: dict[str, Any],
        api_key: str,
        timeout_s: int,
        max_retries: int,
    ) -> tuple[dict[str, Any] | None, str | None]:
        url = f"{self.api_base}/v1beta/models/{self.model}:generateContent"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error = "unknown error"
        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    response = await client.post(url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                last_error = str(e)
                if attempt < max_retries:
                    await asyncio.sleep(min(2 ** (attempt + 1), 60))
                    continue
                return None, f"Error: request failed after retries: {last_error}"

            if response.status_code == 200:
                try:
                    return response.json(), None
                except Exception:
                    return None, "Error: invalid JSON response from ikuncode"

            detail = response.text[:500]
            try:
                parsed = response.json()
                detail = json.dumps(parsed, ensure_ascii=False)[:500]
            except Exception:
                pass
            last_error = f"HTTP {response.status_code}: {detail}"

            if response.status_code in self._RETRYABLE_STATUS_CODES and attempt < max_retries:
                await asyncio.sleep(min(2 ** (attempt + 1), 60))
                continue
            return None, f"Error: image generation failed: {last_error}"

        return None, f"Error: image generation failed: {last_error}"

    @staticmethod
    def _extract_image(data: dict[str, Any]) -> tuple[bytes, str, str | None]:
        try:
            candidates = data.get("candidates") or []
            parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
            image_part = next(
                part for part in parts if isinstance(part, dict) and ("inlineData" in part or "inline_data" in part)
            )
            inline = image_part.get("inlineData") or image_part.get("inline_data") or {}
            b64_data = inline.get("data")
            mime_type = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            if not isinstance(b64_data, str) or not b64_data:
                return b"", "", "Error: image data missing in ikuncode response"
            return base64.b64decode(b64_data), str(mime_type), None
        except Exception:
            snippet = json.dumps(data, ensure_ascii=False)[:500]
            return b"", "", f"Error: failed to parse image from response: {snippet}"

    def _resolve_output_path(self, output_path: str | None, prompt: str, mime_type: str) -> Path:
        ext = "." + mime_type.split("/")[-1].replace("jpeg", "jpg")
        if output_path:
            out = Path(output_path)
            if not out.is_absolute():
                out = self.workspace / out
            if not out.suffix:
                out = out.with_suffix(ext)
            return out

        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        slug = self._prompt_slug(prompt)
        return self.workspace / "outimage" / "ikunimage" / f"{stamp}_{slug}{ext}"

    @staticmethod
    def _prompt_slug(prompt: str) -> str:
        head = (prompt or "").strip().splitlines()[0] if prompt else "image"
        head = re.sub(r"\s+", "_", head)
        head = re.sub(r"[^\w\u4e00-\u9fff]+", "", head)
        if not head:
            return "image"
        return head[:20]
