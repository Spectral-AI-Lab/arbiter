"""Dynamic asynchronous batch scheduler for ARBITER inference serving.

Implements micro-batching across concurrent asynchronous HTTP requests,
bounding latency windows, token counts, and queue depth.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..modeling_arbiter import ArbiterModel
from .metrics import ServingMetrics

logger = logging.getLogger("arbiter.serve.scheduler")


class QueueFullError(Exception):
    """Raised when the scheduler queue exceeds capacity."""
    pass


@dataclass
class QueuedTask:
    """An individual decision item queued for batched execution."""
    payload: dict[str, Any]
    future: asyncio.Future
    created_at: float
    is_singleton_choice: bool = False
    singleton_key: str = ""


class DynamicBatchScheduler:
    """Async dynamic micro-batch scheduler for ArbiterModel."""

    def __init__(
        self,
        model: ArbiterModel,
        metrics: Optional[ServingMetrics] = None,
        batch_window_ms: float = 8.0,
        max_batch_size: int = 16,
        max_batch_tokens: int = 8192,
        max_attention_elements: int = 16777216,
        max_queue_size: int = 1024,
    ):
        self.model = model
        self.metrics = metrics or ServingMetrics(max_history=2000)
        self.batch_window_ms = float(batch_window_ms)
        self.max_batch_size = int(max_batch_size)
        self.max_batch_tokens = int(max_batch_tokens)
        self.max_attention_elements = int(max_attention_elements)
        self.max_queue_size = int(max_queue_size)

        self._queue: asyncio.Queue[Optional[QueuedTask]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._active_batch: list[QueuedTask] = []
        self._stopping = False

    async def start(self) -> None:
        """Start the background worker loop."""
        if self._worker_task is None or self._worker_task.done():
            self._stopping = False
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info(
                "DynamicBatchScheduler started (batch_window=%.1fms, max_batch=%d, max_tokens=%d, max_attn=%d)",
                self.batch_window_ms,
                self.max_batch_size,
                self.max_batch_tokens,
                self.max_attention_elements,
            )

    async def stop(self, drain_timeout: float = 5.0) -> None:
        """Stop background processing gracefully, waiting for current active batch and cancelling pending items."""
        self._stopping = True

        # Drain unstarted pending queue items and cancel their futures cleanly
        while not self._queue.empty():
            try:
                task = self._queue.get_nowait()
                if task is not None and not task.future.done():
                    task.future.set_exception(asyncio.CancelledError("Server shutting down"))
            except asyncio.QueueEmpty:
                break

        # Put sentinel None to unblock worker if waiting at queue.get()
        self._queue.put_nowait(None)

        # Await the current active worker task so any ongoing in-flight batch completes
        if self._worker_task and not self._worker_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._worker_task), timeout=drain_timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
            finally:
                # Guarantee that any task in the active batch is settled
                if self._active_batch:
                    for task in self._active_batch:
                        if not task.future.done():
                            task.future.set_exception(asyncio.CancelledError("Server shut down before batch finished"))
                    self._active_batch = []
            self._worker_task = None
        logger.info("DynamicBatchScheduler stopped.")

    def check_capacity(self, num_slots: int = 1) -> None:
        """Check whether the queue can accept num_slots tasks without overflowing."""
        current_depth = self._queue.qsize()
        if current_depth + num_slots > self.max_queue_size:
            raise QueueFullError(
                f"Scheduler queue is near capacity ({current_depth}/{self.max_queue_size}). Cannot admit {num_slots} questions."
            )

    def submit(
        self,
        payload: dict[str, Any],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> asyncio.Future:
        """Submit a single decision task to the scheduler.
        
        Validates task synchronously before enqueuing to prevent poisoned batches.
        Raises ValueError/TypeError immediately on invalid inputs or budget overflow.
        """
        loop = loop or asyncio.get_running_loop()
        future = loop.create_future()

        # Handle singleton choice bypass (trivially 1.0 confidence)
        if payload.get("_is_singleton_choice"):
            singleton_key = payload.get("_singleton_key", "A")
            crit = payload.get("criteria", {})
            if isinstance(crit, dict):
                cand_desc = crit.get(singleton_key)
            elif isinstance(crit, (list, tuple)) and len(crit) > 0:
                cand_desc = crit[0]
            else:
                cand_desc = None

            clean_ctx = re.sub(r"<OPTION>", "[OPTION]", str(payload.get("context", "")).strip())
            clean_inst = re.sub(r"<OPTION>", "[OPTION]", str(payload.get("instruction", "")).strip())
            opt_line = f"<OPTION> {singleton_key}: {re.sub(r'<OPTION>', '[OPTION]', str(cand_desc).strip())}" if cand_desc else f"<OPTION> {singleton_key}"
            options_text = opt_line
            header = f"Contexte : {clean_ctx}\nInstruction : {clean_inst}\n\n"
            prompt = header + options_text

            input_tokens = 1
            is_truncated = False
            max_length = payload.get("max_length", 4096)
            allow_truncation = payload.get("allow_truncation", False)

            if self.model.tokenizer:
                ids = self.model.tokenizer.encode(prompt, add_special_tokens=True)
                raw_token_count = len(ids)
                if raw_token_count > max_length:
                    if not allow_truncation:
                        raise ValueError(
                            f"Input sequence length ({raw_token_count} tokens) exceeds max_length ({max_length})."
                        )
                    ctx_ids = self.model.tokenizer.encode(clean_ctx, add_special_tokens=False)
                    while True:
                        trunc_ctx = self.model.tokenizer.decode(ctx_ids, skip_special_tokens=True)
                        new_prompt = f"Contexte : {trunc_ctx}\nInstruction : {clean_inst}\n\n{options_text}"
                        ids = self.model.tokenizer.encode(new_prompt, add_special_tokens=True)
                        if len(ids) <= max_length or len(ctx_ids) == 0:
                            break
                        overshoot = max(1, len(ids) - max_length)
                        ctx_ids = ctx_ids[:-overshoot]
                    is_truncated = True
                input_tokens = len(ids)

            # Check budgets for singleton too
            if input_tokens > self.max_batch_tokens or (input_tokens * input_tokens) > self.max_attention_elements:
                raise ValueError(
                    f"Request length ({input_tokens} tokens) exceeds maximum batch tokens ({self.max_batch_tokens}) or attention budget ({self.max_attention_elements})."
                )

            # Record shortcut decision in metrics
            self.metrics.record_batch(1, 0.1, input_tokens)

            future.set_result({
                "selected_key": singleton_key,
                "probabilities": {singleton_key: 1.0},
                "confidence": 1.0,
                "input_tokens": input_tokens,
                "truncated": is_truncated,
                "_is_singleton_choice": True,
                "_singleton_key": singleton_key,
            })
            return future

        # Clean payload for model execution
        exec_payload = {
            "context": payload["context"],
            "instruction": payload["instruction"],
            "criteria": payload["criteria"],
            "question_type": payload.get("question_type", "choice"),
            "max_length": payload.get("max_length", 4096),
            "allow_truncation": payload.get("allow_truncation", False),
        }

        # Validate task preparation and token budget synchronously before queueing
        # Raises ValueError/TypeError directly if invalid
        prep = self.model._prepare_decision(**exec_payload)
        seq_len = len(prep["ids"])
        if seq_len > self.max_batch_tokens or (seq_len * seq_len) > self.max_attention_elements:
            raise ValueError(
                f"Request length ({seq_len} tokens) exceeds maximum batch tokens ({self.max_batch_tokens}) or attention budget ({self.max_attention_elements})."
            )

        # Cache prepared decision to prevent double tokenization in decide_batch
        exec_payload["_prepared"] = prep

        # Check queue depth
        q_size = self._queue.qsize()
        if q_size >= self.max_queue_size:
            raise QueueFullError(
                f"Scheduler queue is full ({q_size}/{self.max_queue_size} pending items)."
            )

        task = QueuedTask(
            payload=exec_payload,
            future=future,
            created_at=time.monotonic(),
        )

        self._queue.put_nowait(task)
        self.metrics.update_queue_depth(self._queue.qsize(), self.max_queue_size)
        return future

    async def _worker_loop(self) -> None:
        """Async worker loop collecting micro-batches."""
        while not self._stopping:
            try:
                first_task = await self._queue.get()
            except asyncio.CancelledError:
                break

            if first_task is None:
                # Sentinel to stop worker
                break

            batch = [first_task]
            deadline = time.monotonic() + (self.batch_window_ms / 1000.0)

            # Micro-batch collection window
            if self.batch_window_ms > 0:
                while len(batch) < self.max_batch_size:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        next_task = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                        if next_task is None:
                            break
                        batch.append(next_task)
                    except asyncio.TimeoutError:
                        break
                    except asyncio.CancelledError:
                        break
            else:
                # Immediate dispatch: drain whatever is available right now
                while len(batch) < self.max_batch_size and not self._queue.empty():
                    try:
                        next_task = self._queue.get_nowait()
                        if next_task is None:
                            break
                        batch.append(next_task)
                    except asyncio.QueueEmpty:
                        break

            self.metrics.update_queue_depth(self._queue.qsize(), self.max_queue_size)

            # Execute batch
            await self._dispatch_batch(batch)

    async def _dispatch_batch(self, batch: list[QueuedTask]) -> None:
        """Dispatch a gathered batch to the PyTorch inference engine."""
        self._active_batch = batch
        try:
            # Filter out cancelled tasks before running inference
            active_batch = [t for t in batch if not t.future.cancelled()]
            if not active_batch:
                return

            requests_payload = [t.payload for t in active_batch]
            start_time = time.monotonic()

            try:
                results = await asyncio.to_thread(
                    self.model.decide_batch,
                    requests_payload,
                    batch_size=self.max_batch_size,
                    max_batch_tokens=self.max_batch_tokens,
                    max_attention_elements=self.max_attention_elements,
                )
                duration_ms = (time.monotonic() - start_time) * 1000.0
                total_tokens = sum(r.get("input_tokens", 0) for r in results)
                self.metrics.record_batch(len(active_batch), duration_ms, total_tokens)

                for task, res in zip(active_batch, results):
                    if not task.future.done():
                        task.future.set_result(res)

            except Exception as exc:
                logger.error("Error executing batched inference: %s", exc, exc_info=True)
                for task in active_batch:
                    if not task.future.done():
                        task.future.set_exception(exc)
        finally:
            # Safety settlement: ensure no future in batch is left pending
            for task in batch:
                if not task.future.done():
                    task.future.set_exception(asyncio.CancelledError("Inference task was interrupted"))
            self._active_batch = []
