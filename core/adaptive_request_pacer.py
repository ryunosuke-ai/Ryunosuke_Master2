"""並列API呼び出しの開始間隔を均等化する共有ペーサー。"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable


class AdaptiveRequestPacer:
    """全workerで呼び出し間隔と429後のcooldownを共有する。"""

    def __init__(
        self,
        *,
        requests_per_minute: float = 0.0,
        minimum_interval_seconds: float = 0.0,
        initial_backoff_seconds: float = 15.0,
        maximum_interval_seconds: float = 5.0,
        recovery_successes: int = 20,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        rpm_interval = (
            60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        )
        self.base_interval = max(0.0, minimum_interval_seconds, rpm_interval)
        self.current_interval = self.base_interval
        self.initial_backoff_seconds = max(0.0, initial_backoff_seconds)
        self.maximum_interval_seconds = max(
            self.base_interval, maximum_interval_seconds
        )
        self.recovery_successes = max(1, recovery_successes)
        self._clock = clock
        self._sleep = sleeper
        self._jitter = jitter
        self._lock = threading.Lock()
        self._next_request_at = 0.0
        self._blocked_until = 0.0
        self._successes_since_limit = 0

    @property
    def enabled(self) -> bool:
        """開始間隔の制御が有効か返す。"""
        return self.base_interval > 0

    def wait(self) -> None:
        """共有スケジュール上の次のAPI開始時刻まで待つ。"""
        if not self.enabled:
            return
        while True:
            with self._lock:
                now = self._clock()
                scheduled_at = max(self._next_request_at, self._blocked_until)
                delay = scheduled_at - now
                if delay <= 0:
                    self._next_request_at = now + self.current_interval
                    return
            self._sleep(delay)

    def record_rate_limit(self, retry_after_seconds: float | None = None) -> float:
        """429を全workerへ反映し、次の試行までの待機秒数を返す。"""
        with self._lock:
            now = self._clock()
            self.current_interval = min(
                self.maximum_interval_seconds,
                max(self.base_interval, self.current_interval * 1.5),
            )
            delay = max(
                self.initial_backoff_seconds,
                float(retry_after_seconds or 0.0),
            )
            delay += self._jitter(0.0, min(1.0, delay * 0.1))
            self._blocked_until = max(self._blocked_until, now + delay)
            self._next_request_at = max(self._next_request_at, self._blocked_until)
            self._successes_since_limit = 0
            return delay

    def record_success(self) -> None:
        """成功が続いた場合、429前の効率へ緩やかに戻す。"""
        if not self.enabled:
            return
        with self._lock:
            self._successes_since_limit += 1
            if self._successes_since_limit < self.recovery_successes:
                return
            self.current_interval = max(
                self.base_interval, self.current_interval * 0.9
            )
            self._successes_since_limit = 0
