"""Metrics collector for ARBITER inference serving."""

import collections
import statistics
import threading
import time
from typing import Optional


class ServingMetrics:
    """Thread-safe operational metrics collector."""

    def __init__(self, max_history: int = 2000):
        self._lock = threading.Lock()
        self._max_history = max_history
        self.total_requests = 0
        self.total_decisions = 0
        self.total_tokens = 0
        self._latencies_ms = collections.deque(maxlen=max_history)
        self._batch_sizes = collections.deque(maxlen=max_history)
        self._queue_size = 0
        self._max_queue_size = 1024

    def update_queue_depth(self, current_size: int, max_size: int) -> None:
        with self._lock:
            self._queue_size = current_size
            self._max_queue_size = max_size

    def record_batch(self, batch_size: int, duration_ms: float, tokens: int) -> None:
        with self._lock:
            self.total_decisions += batch_size
            self.total_tokens += tokens
            self._batch_sizes.append(batch_size)

    def record_request(self, duration_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            self._latencies_ms.append(duration_ms)

    def snapshot(self) -> dict:
        with self._lock:
            latencies = list(self._latencies_ms)
            batches = list(self._batch_sizes)
            q_size = self._queue_size
            max_q = self._max_queue_size

        if latencies:
            lat_sorted = sorted(latencies)
            n = len(lat_sorted)
            p50 = lat_sorted[int(n * 0.50)]
            p95 = lat_sorted[min(int(n * 0.95), n - 1)]
        else:
            p50 = 0.0
            p95 = 0.0

        avg_batch = statistics.mean(batches) if batches else 0.0
        saturation = round(q_size / max_q, 4) if max_q > 0 else 0.0

        return dict(
            total_requests=self.total_requests,
            total_decisions=self.total_decisions,
            total_tokens=self.total_tokens,
            avg_batch_size=round(avg_batch, 2),
            p50_latency_ms=round(p50, 2),
            p95_latency_ms=round(p95, 2),
            queue_size=q_size,
            queue_saturation=saturation,
        )
