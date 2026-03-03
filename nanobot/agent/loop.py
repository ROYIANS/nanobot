"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import weakref
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.ikunimage import IkunImageTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, ProviderConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 500
    _TASK_META_KEY = "active_task"
    _MAX_ITER_MSG_PREFIX = "I reached the maximum number of tool call iterations"
    _ADMIN_ONLY_TOOLS = {"write_file", "edit_file", "exec", "spawn"}
    _IN_PROGRESS_PATTERNS = (
        re.compile(
            r"\b(i(?:'|’)ll|i will|next[, ]+i(?:'|’)ll|i'm going to|let me (?:check|look|inspect|review|fix|do|work on|continue|investigate))\b",
            re.IGNORECASE,
        ),
        re.compile(r"(我先|我会先|接下来我会|下一步我会|稍后我会|让我先)"),
    )

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        ikuncode_config: ProviderConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.ikuncode_config = ikuncode_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=reasoning_effort,
            brave_api_key=brave_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            ikuncode_config=ikuncode_config,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._archive_tasks: set[asyncio.Task] = set()  # Strong refs to /new archival tasks
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(
            IkunImageTool(
                workspace=self.workspace,
                api_key=(self.ikuncode_config.api_key if self.ikuncode_config else None),
                api_base=(self.ikuncode_config.api_base if self.ikuncode_config else None),
            )
        )
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat()

    def _get_active_task(self, session: Session) -> dict[str, Any] | None:
        metadata = getattr(session, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        task = metadata.get(self._TASK_META_KEY)
        return task if isinstance(task, dict) else None

    def _set_active_task(self, session: Session, task: dict[str, Any]) -> None:
        metadata = getattr(session, "metadata", None)
        if not isinstance(metadata, dict):
            return
        metadata[self._TASK_META_KEY] = task
        session.updated_at = datetime.now()

    def _archive_snapshot_after_new(self, session_key: str, snapshot: list[dict[str, Any]]) -> None:
        """Archive a copied snapshot in background so /new can return immediately."""
        if not snapshot:
            return

        async def _run() -> None:
            try:
                temp = Session(key=session_key)
                temp.messages = list(snapshot)
                ok = await self._consolidate_memory(temp, archive_all=True)
                if not ok:
                    logger.warning("/new background archival failed for {}", session_key)
            except Exception:
                logger.exception("/new background archival crashed for {}", session_key)
            finally:
                task = asyncio.current_task()
                if task is not None:
                    self._archive_tasks.discard(task)

        task = asyncio.create_task(_run())
        self._archive_tasks.add(task)

    def _build_continue_prompt(self, task: dict[str, Any]) -> str:
        objective = str(task.get("objective") or "").strip() or "(unknown)"
        last_message = str(task.get("last_assistant_message") or "").strip() or "(none)"
        return (
            "Continue the active task from where you left off.\n"
            f"Original task: {objective}\n"
            f"Last assistant update: {last_message}\n"
            "Do not restart from scratch. Reuse prior progress and finish the task now."
        )

    def _infer_task_status(self, final_content: str) -> tuple[str, str]:
        content = (final_content or "").strip()
        lowered = content.lower()

        if not content:
            return "in_progress", "empty_response"
        if self._MAX_ITER_MSG_PREFIX in content:
            return "in_progress", "max_iterations"
        if "encountered an error" in lowered or lowered.startswith("error calling llm"):
            return "blocked", "error_response"
        for pattern in self._IN_PROGRESS_PATTERNS:
            if pattern.search(content):
                return "in_progress", "future_intent"
        return "completed", "final_response"

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        allowed_tool_names: set[str] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(allowed_names=allowed_tool_names),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(
                        tool_call.name,
                        tool_call.arguments,
                        allowed_names=allowed_tool_names,
                    )
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    def _allowed_tool_names_for_message(self, msg: InboundMessage) -> set[str] | None:
        """Return tool allowlist for this message, or None for full access."""
        if msg.channel != "feishu":
            return None
        if not self.channels_config:
            return None

        feishu_cfg = getattr(self.channels_config, "feishu", None)
        if not feishu_cfg:
            return None

        admin_ids = {str(x).strip() for x in (feishu_cfg.admin_ids or []) if str(x).strip()}
        if not admin_ids:
            return None

        sender = str((msg.metadata or {}).get("group_sender_id") or msg.sender_id or "").strip()
        if sender in admin_ids:
            return None

        all_tools = set(self.tools.tool_names)
        return all_tools - self._ADMIN_ONLY_TOOLS

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        try:
            session = self.sessions.get_or_create(msg.session_key)
            task = self._get_active_task(session)
            if task:
                task["status"] = "cancelled"
                task["status_reason"] = "stopped_by_user"
                task["updated_at"] = self._now_iso()
                self._set_active_task(session, task)
                self.sessions.save(session)
        except Exception:
            logger.debug("Skip active task update during /stop for {}", msg.session_key)
        metadata = dict(msg.metadata or {})
        if msg.channel == "feishu" and metadata.get("message_id"):
            metadata["_turn_done"] = True
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=metadata,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    if msg.channel == "feishu" and msg.metadata.get("message_id"):
                        meta = dict(response.metadata or {})
                        meta.setdefault("message_id", msg.metadata.get("message_id"))
                        meta["_turn_done"] = True
                        response.metadata = meta
                    await self.bus.publish_outbound(response)
                elif msg.channel == "feishu" and msg.metadata.get("message_id"):
                    meta = dict(msg.metadata or {})
                    meta["_turn_done"] = True
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=meta,
                    ))
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                metadata = {}
                if msg.channel == "feishu" and msg.metadata.get("message_id"):
                    metadata = dict(msg.metadata or {})
                    metadata["_turn_done"] = True
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.", metadata=metadata,
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())
            self._consolidating.add(session.key)
            snapshot: list[dict[str, Any]] = []
            try:
                async with lock:
                    snapshot = list(session.messages[session.last_consolidated:])
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
            finally:
                self._consolidating.discard(session.key)

            session.clear()
            session.metadata = {}
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            self._archive_snapshot_after_new(session.key, snapshot)
            if msg.channel == "feishu":
                meta = dict(msg.metadata or {})
                meta["feishu_msg_type"] = "system"
                meta["feishu_system_content"] = {
                    "type": "divider",
                    "params": {
                        "divider_text": {
                            "text": "新会话",
                            "i18n_text": {
                                "zh_CN": "新会话",
                                "en_US": "New Session",
                            },
                        },
                    },
                    "options": {
                        "need_rollup": True,
                    },
                }
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id, content="", metadata=meta
                )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/continue":
            task = self._get_active_task(session)
            if not task or task.get("status") in {"completed", "cancelled"}:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="No active task to continue. Send a new request first.",
                )
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/continue — Continue the current task\n/stop — Stop the current task\n/help — Show available commands")
        if cmd == "/botid":
            bot_id = str(
                (msg.metadata or {}).get("bot_open_id")
                or (
                    getattr(getattr(self.channels_config, "feishu", None), "bot_open_id", "")
                    if self.channels_config else ""
                )
                or ""
            ).strip()
            if bot_id:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Bot open_id: {bot_id}",
                )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    "Bot open_id is not resolved yet. "
                    "Please @bot once in group, or set channels.feishu.botOpenId in config."
                ),
            )

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        effective_message = msg.content
        objective = msg.content
        if cmd == "/continue":
            task = self._get_active_task(session) or {}
            objective = str(task.get("objective") or "").strip() or msg.content
            effective_message = self._build_continue_prompt(task)
            task["status"] = "running"
            task["status_reason"] = "resumed"
            task["objective"] = objective
            task["updated_at"] = self._now_iso()
            task["resume_count"] = int(task.get("resume_count") or 0) + 1
            self._set_active_task(session, task)
            self.sessions.save(session)
        elif not cmd.startswith("/"):
            self._set_active_task(
                session,
                {
                    "status": "running",
                    "status_reason": "new_request",
                    "objective": objective,
                    "created_at": self._now_iso(),
                    "updated_at": self._now_iso(),
                    "resume_count": 0,
                    "turn_count": 0,
                },
            )
            self.sessions.save(session)

        # Preserve group speaker identity in shared room sessions.
        if (
            not cmd.startswith("/")
            and bool((msg.metadata or {}).get("is_group"))
            and str((msg.metadata or {}).get("group_sender_id") or "").strip()
        ):
            speaker = str((msg.metadata or {}).get("group_sender_id")).strip()
            effective_message = f"[Group speaker: {speaker}]\n{msg.content}"

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=effective_message,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        allowed_tool_names = self._allowed_tool_names_for_message(msg)
        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            allowed_tool_names=allowed_tool_names,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        task = self._get_active_task(session)
        if task:
            status, reason = self._infer_task_status(final_content)
            task["status"] = status
            task["status_reason"] = reason
            task["objective"] = objective
            task["last_assistant_message"] = final_content
            task["updated_at"] = self._now_iso()
            task["turn_count"] = int(task.get("turn_count") or 0) + 1
            self._set_active_task(session, task)
        self.sessions.save(session)

        message_tool = self.tools.get("message")
        if (
            msg.channel == "feishu"
            and isinstance(message_tool, MessageTool)
            and message_tool._sent_in_turn
        ):
            task_status = str((task or {}).get("status") or "")
            if task_status == "completed" or not task_status:
                logger.info(
                    "Suppressing final Feishu text reply for {}:{} because same-target message tool already sent content",
                    msg.channel,
                    msg.sender_id,
                )
                return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    continue
                if isinstance(content, list):
                    entry["content"] = [
                        {"type": "text", "text": "[image]"} if (
                            c.get("type") == "image_url"
                            and c.get("image_url", {}).get("url", "").startswith("data:image/")
                        ) else c for c in content
                    ]
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        return await MemoryStore(self.workspace).consolidate(
            session, self.provider, self.model,
            archive_all=archive_all, memory_window=self.memory_window,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
