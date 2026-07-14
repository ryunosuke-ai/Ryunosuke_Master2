from __future__ import annotations

from core.adaptive_request_pacer import AdaptiveRequestPacer
from tools.score_dialogue_with_bayes_model import _retry_after_seconds


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_pacer_evenly_spaces_requests_and_recovers_after_429():
    clock = FakeClock()
    pacer = AdaptiveRequestPacer(
        requests_per_minute=120,
        initial_backoff_seconds=15,
        recovery_successes=2,
        clock=clock.monotonic,
        sleeper=clock.sleep,
        jitter=lambda _low, _high: 0.0,
    )

    pacer.wait()
    pacer.wait()
    assert clock.sleeps == [0.5]

    delay = pacer.record_rate_limit(retry_after_seconds=20)
    assert delay == 20
    assert pacer.current_interval == 0.75
    pacer.wait()
    assert clock.now == 20.5

    pacer.record_success()
    pacer.record_success()
    assert pacer.current_interval == 0.675


def test_retry_after_ms_takes_precedence():
    class Response:
        headers = {"retry-after-ms": "2500", "retry-after": "9"}

    class Error(Exception):
        response = Response()

    assert _retry_after_seconds(Error()) == 2.5


def test_disabled_pacer_keeps_legacy_behavior():
    clock = FakeClock()
    pacer = AdaptiveRequestPacer(
        clock=clock.monotonic,
        sleeper=clock.sleep,
    )
    pacer.wait()
    assert not pacer.enabled
    assert clock.sleeps == []
