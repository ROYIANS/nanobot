"""Ikun image generation tool backed by ikuncode NanoBananaPro API.

Mirrors tmp/IKunImage-main skill exactly:
  - API:     https://api.ikuncode.cc
  - Model:   gemini-3-pro-image-preview
  - Key:     IKUN_API_KEY > ~/.ikunimage/config.json
  - Output:  outimage/ikunimage/{YYYYMMDD}_{HHMM}_{主题简称}.png
  - Retry:   exponential backoff min(2**n, 60)s
  - Batch:   async concurrent via batch_json parameter
"""

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

# ---------------------------------------------------------------------------
# 常量（与 generate_ikun.py 完全一致）
# ---------------------------------------------------------------------------

BASE_URL = "https://api.ikuncode.cc"
MODEL_PATH = "/v1beta/models/gemini-3-pro-image-preview:generateContent"
CONFIG_FILE = Path.home() / ".ikunimage" / "config.json"

VALID_ASPECT_RATIOS = (
    "1:1", "16:9", "9:16", "4:3", "3:4",
    "3:2", "2:3", "21:9", "5:4", "4:5",
)
VALID_SIZES = ("1K", "2K", "4K")
TIMEOUT_BY_SIZE = {"1K": 360, "2K": 600, "4K": 1200}
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_EDIT_INPUT_BYTES = 4 * 1024 * 1024
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


class IkunImageTool(Tool):
    """Image generation tool for ikuncode (NanoBananaPro / Gemini image model).

    Supports text-to-image, image editing, and concurrent batch generation.
    Style and language are fully user-controlled.
    """

    def __init__(
        self,
        workspace: Path,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str = "gemini-3-pro-image-preview",
    ) -> None:
        self.workspace = workspace
        self._api_key = (api_key or "").strip()
        self.api_base = (api_base or BASE_URL).rstrip("/")
        self.model = (model or "gemini-3-pro-image-preview").strip()

    @property
    def name(self) -> str:
        return "ikun_image"

    @property
    def description(self) -> str:
        return (
            "Generate or edit images via ikuncode (NanoBananaPro model). "
            "Supports text-to-image, image editing, and concurrent batch generation. "
            "Style and language are user-controlled; follow the user's prompt as-is. "
            "Outputs to outimage/ikunimage/{YYYYMMDD}_{HHMM}_{slug}.png by default."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Image prompt (free style, any language). Optional in batch mode.",
                },
                "input_image": {
                    "type": "string",
                    "description": (
                        "图生图模式：本地输入图片路径（JPG/PNG/WebP/GIF，< 4MB）。"
                        "提供此参数时走图生图流程，size 参数不生效。"
                    ),
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": list(VALID_ASPECT_RATIOS),
                    "description": "输出宽高比，默认 1:1。竖版→9:16，横版→16:9，超宽→21:9。",
                },
                "size": {
                    "type": "string",
                    "enum": list(VALID_SIZES),
                    "description": "分辨率等级（仅文生图）：1K 快速预览 / 2K 推荐 / 4K 超高清。默认 2K。",
                },
                "output_path": {
                    "type": "string",
                    "description": "可选输出路径；相对路径基于工作区。留空则按规范自动命名。",
                },
                "batch_json": {
                    "type": "string",
                    "description": (
                        "批量生成：JSON 数组字符串，每项为 "
                        "{\"prompt\": ..., \"aspect_ratio\"?: ..., \"size\"?: ..., \"output\"?: ..., \"input\"?: ...}。"
                        "含 input 字段时走图生图；否则文生图。"
                    ),
                },
                "workers": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "description": "批量并发数，默认 2。",
                },
                "retry": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "每个任务最大重试次数，默认 3。",
                },
                "api_key": {
                    "type": "string",
                    "description": "可选 API Key 覆盖（优先级最高）。",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        prompt: str | None = None,
        input_image: str | None = None,
        aspect_ratio: str = "1:1",
        size: str = "2K",
        output_path: str | None = None,
        batch_json: str | None = None,
        workers: int = 2,
        retry: int = 3,
        api_key: str | None = None,
        **_: Any,
    ) -> str:
        key = self._resolve_api_key(api_key)
        if not key:
            skills_dir = "~/.nanobot/skills/ikunimage/scripts"
            return (
                "错误：未找到 ikun API Key。请通过以下方式之一配置：\n"
                f"  1. 运行 python {skills_dir}/generate_ikun.py --setup\n"
                f"  2. 手动创建 {CONFIG_FILE}，内容：{{\"api_key\": \"sk-xxx\"}}\n"
                "  3. 设置环境变量 IKUN_API_KEY=sk-xxx\n"
                "  4. 传入 api_key 参数"
            )

        # ── 批量模式 ──────────────────────────────────────────────────────
        if batch_json:
            return await self._run_batch(
                batch_json=batch_json,
                api_key=key,
                workers=workers,
                retry=retry,
            )

        # ── 单图模式 ──────────────────────────────────────────────────────
        if not prompt or not prompt.strip():
            return "错误：必须提供 prompt（单图模式）或 batch_json（批量模式）"

        prompt = prompt.strip()

        aspect_ratio = (aspect_ratio or "1:1").strip()
        if aspect_ratio not in VALID_ASPECT_RATIOS:
            return f"错误：无效的 aspect_ratio。支持：{', '.join(VALID_ASPECT_RATIOS)}"

        size = (size or "2K").strip().upper()
        if size not in VALID_SIZES:
            return "错误：无效的 size。支持：1K、2K、4K"

        result = await self._generate_one(
            prompt=prompt,
            input_image=input_image,
            aspect_ratio=aspect_ratio,
            size=size,
            output_path=output_path,
            api_key=key,
            max_retries=max(0, retry),
        )
        if not result["success"]:
            return f"错误：{result['error']}"

        return (
            f"图片已生成：{result['path']} "
            f"({result['size_kb']:.0f} KB，耗时 {result['elapsed']:.1f}s)"
        )

    # ──────────────────────────────────────────────────────────────────────
    # API Key 解析（优先级：参数 > IKUN_API_KEY 环境变量 > 配置文件）
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_api_key(self, override: str | None) -> str:
        if override and override.strip():
            return override.strip()
        if self._api_key:
            return self._api_key
        env = os.environ.get("IKUN_API_KEY", "").strip()
        if env:
            return env
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                key = str((data or {}).get("api_key") or "").strip()
                if key:
                    return key
            except Exception:
                pass
        return ""

    # ──────────────────────────────────────────────────────────────────────
    # 单张生成核心
    # ──────────────────────────────────────────────────────────────────────

    async def _generate_one(
        self,
        prompt: str,
        input_image: str | None,
        aspect_ratio: str,
        size: str,
        output_path: str | None,
        api_key: str,
        max_retries: int = 3,
    ) -> dict:
        payload, err = self._build_payload(prompt, input_image, aspect_ratio, size)
        if err:
            return {"success": False, "error": err, "path": "", "size_kb": 0.0, "elapsed": 0.0}

        timeout = TIMEOUT_BY_SIZE.get(size, 600) if not input_image else 600
        data, err, elapsed = await self._request_with_retry(payload, api_key, timeout, max_retries)
        if err:
            return {"success": False, "error": err, "path": "", "size_kb": 0.0, "elapsed": 0.0}

        image_bytes, mime_type, err = self._extract_image(data)
        if err:
            return {"success": False, "error": err, "path": "", "size_kb": 0.0, "elapsed": 0.0}

        destination = self._resolve_output_path(output_path, prompt, mime_type)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)

        return {
            "success": True,
            "path": str(destination.resolve()),
            "size_kb": round(len(image_bytes) / 1024, 1),
            "elapsed": elapsed,
        }

    # ──────────────────────────────────────────────────────────────────────
    # 批量生成（并发，Semaphore 控制 workers 数）
    # ──────────────────────────────────────────────────────────────────────

    async def _run_batch(
        self,
        batch_json: str,
        api_key: str,
        workers: int,
        retry: int,
    ) -> str:
        try:
            tasks = json.loads(batch_json)
        except json.JSONDecodeError as e:
            return f"错误：batch_json 解析失败：{e}"

        if not isinstance(tasks, list) or not tasks:
            return "错误：batch_json 必须是非空 JSON 数组"

        for i, t in enumerate(tasks):
            if not isinstance(t, dict):
                return f"错误：任务 #{i + 1} 格式错误，必须是 JSON 对象"
            if not t.get("prompt"):
                return f"错误：任务 #{i + 1} 缺少必填字段 prompt"
            if t.get("input") and not Path(str(t["input"])).exists() and not (self.workspace / t["input"]).exists():
                return f"错误：任务 #{i + 1} 输入图片不存在：{t['input']}"

        num = len(tasks)
        effective_workers = max(1, min(workers, num, 8))
        sem = asyncio.Semaphore(effective_workers)

        async def _run(index: int, task: dict) -> tuple[int, dict]:
            async with sem:
                return index, await self._generate_one(
                    prompt=task["prompt"],
                    input_image=task.get("input"),
                    aspect_ratio=task.get("aspect_ratio", "1:1"),
                    size=task.get("size", "2K"),
                    output_path=task.get("output"),
                    api_key=api_key,
                    max_retries=retry,
                )

        results: dict[int, dict] = {}
        for coro in asyncio.as_completed([_run(i, t) for i, t in enumerate(tasks)]):
            idx, result = await coro
            results[idx] = result

        ordered = [results[i] for i in range(num)]
        ok = sum(1 for r in ordered if r["success"])
        lines = [f"[ikunimage 批量] {ok}/{num} 成功"]
        for i, r in enumerate(ordered):
            if r["success"]:
                lines.append(f"  #{i + 1} OK → {r['path']} ({r['size_kb']:.0f} KB，{r['elapsed']:.1f}s)")
            else:
                lines.append(f"  #{i + 1} FAIL → {r['error']}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # Payload 构建（文生图 / 图生图，与脚本完全一致）
    # ──────────────────────────────────────────────────────────────────────

    def _build_payload(
        self,
        prompt: str,
        input_image: str | None,
        aspect_ratio: str,
        size: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if input_image:
            path = Path(input_image)
            if not path.is_absolute():
                path = self.workspace / path
            if not path.exists() or not path.is_file():
                return None, f"图片文件不存在：{input_image}"
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                return None, (
                    f"不支持的图片格式 '{path.suffix}'，"
                    f"支持：{', '.join(sorted(SUPPORTED_IMAGE_EXTS))}"
                )
            if path.stat().st_size > MAX_EDIT_INPUT_BYTES:
                size_mb = path.stat().st_size / 1024 / 1024
                return None, f"图片过大（{size_mb:.1f}MB），建议 < 4MB"
            mime_type, _ = mimetypes.guess_type(str(path))
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime_type, "data": encoded}},
                    ]
                }],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"aspectRatio": aspect_ratio},
                },
            }, None

        # 文生图
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": aspect_ratio, "image_size": size},
            },
        }, None

    # ──────────────────────────────────────────────────────────────────────
    # HTTP 请求（指数退避重试，与脚本逻辑一致）
    # ──────────────────────────────────────────────────────────────────────

    async def _request_with_retry(
        self,
        payload: dict[str, Any],
        api_key: str,
        timeout_s: int,
        max_retries: int,
    ) -> tuple[dict[str, Any] | None, str | None, float]:
        """返回 (data, error, elapsed_seconds)。"""
        import time

        url = f"{self.api_base}{MODEL_PATH}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_error = "未知错误"
        t0 = time.time()

        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = min(2 ** attempt, 60)
                await asyncio.sleep(delay)

            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    resp = await client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException:
                last_error = "请求超时"
                continue
            except (httpx.ConnectError, httpx.NetworkError) as e:
                last_error = f"连接失败：{e}"
                continue

            if resp.status_code == 200:
                elapsed = time.time() - t0
                try:
                    return resp.json(), None, round(elapsed, 1)
                except Exception:
                    return None, "API 返回了无效的 JSON", round(elapsed, 1)

            try:
                detail = json.dumps(resp.json(), ensure_ascii=False)[:500]
            except Exception:
                detail = resp.text[:500]
            last_error = f"HTTP {resp.status_code}：{detail}"

            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                continue
            return None, last_error, round(time.time() - t0, 1)

        return None, f"重试 {max_retries} 次后仍然失败。最后错误：{last_error}", round(time.time() - t0, 1)

    # ──────────────────────────────────────────────────────────────────────
    # 响应解析（与脚本完全一致）
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_image(data: dict[str, Any]) -> tuple[bytes, str, str | None]:
        try:
            parts = data["candidates"][0]["content"]["parts"]
            image_part = next(
                p for p in parts
                if isinstance(p, dict) and ("inlineData" in p or "inline_data" in p)
            )
            inline = image_part.get("inlineData") or image_part.get("inline_data") or {}
            b64_data = inline.get("data")
            mime_type = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            if not isinstance(b64_data, str) or not b64_data:
                return b"", "", "API 响应中图片数据缺失"
            return base64.b64decode(b64_data), str(mime_type), None
        except (KeyError, IndexError, StopIteration):
            snippet = json.dumps(data, ensure_ascii=False)[:500]
            return b"", "", f"API 响应中未找到图片数据：{snippet}"

    # ──────────────────────────────────────────────────────────────────────
    # 输出路径（命名规范：{YYYYMMDD}_{HHMM}_{主题简称}.png）
    # ──────────────────────────────────────────────────────────────────────

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
        return self.workspace / "outimage" / "ikunimage" / f"{stamp}_{slug}.png"

    @staticmethod
    def _prompt_slug(prompt: str) -> str:
        """提取 2-6 个汉字作为主题简称；无汉字则降级为 ASCII slug。"""
        cn = re.findall(r"[\u4e00-\u9fff]", prompt)
        if len(cn) >= 2:
            return "".join(cn[:6])
        head = (prompt or "").strip().splitlines()[0]
        head = re.sub(r"\s+", "_", head)
        head = re.sub(r"[^\w]+", "", head)
        return head[:12] or "image"
