"""Tests for the public-deployment throttling."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.ratelimit import RateLimiter, client_ip


class FakeRequest:
    """Just enough of ``starlette.Request`` for the limiter."""

    def __init__(self, headers: dict[str, str] | None = None, host: str = "1.2.3.4"):
        self.headers = headers or {}
        self.client = type("Client", (), {"host": host})()


def test_cloudflare_header_wins_over_socket_address():
    request = FakeRequest({"cf-connecting-ip": "203.0.113.9"}, host="10.0.0.1")
    assert client_ip(request) == "203.0.113.9"


def test_forwarded_for_takes_the_first_hop():
    request = FakeRequest({"x-forwarded-for": "203.0.113.9, 10.0.0.1"})
    assert client_ip(request) == "203.0.113.9"


def test_falls_back_to_the_socket_address():
    assert client_ip(FakeRequest(host="192.0.2.7")) == "192.0.2.7"


def test_per_ip_window_blocks_the_next_request():
    limiter = RateLimiter(per_ip=2, window_seconds=300, daily_total=0)
    request = FakeRequest()
    limiter.check(request)
    limiter.check(request)
    with pytest.raises(HTTPException) as raised:
        limiter.check(request)
    assert raised.value.status_code == 429
    assert "Retry-After" in raised.value.headers


def test_clients_are_limited_independently():
    limiter = RateLimiter(per_ip=1, window_seconds=300, daily_total=0)
    limiter.check(FakeRequest({"cf-connecting-ip": "203.0.113.1"}))
    limiter.check(FakeRequest({"cf-connecting-ip": "203.0.113.2"}))  # must not raise


def test_daily_budget_stops_everyone():
    limiter = RateLimiter(per_ip=0, window_seconds=300, daily_total=2)
    limiter.check(FakeRequest({"cf-connecting-ip": "203.0.113.1"}))
    limiter.check(FakeRequest({"cf-connecting-ip": "203.0.113.2"}))
    with pytest.raises(HTTPException) as raised:
        limiter.check(FakeRequest({"cf-connecting-ip": "203.0.113.3"}))
    assert raised.value.status_code == 429
    assert "daily" in raised.value.detail.lower()


def test_zero_disables_a_limit():
    limiter = RateLimiter(per_ip=0, window_seconds=300, daily_total=0)
    request = FakeRequest()
    for _ in range(50):
        limiter.check(request)


def test_snapshot_reports_usage():
    limiter = RateLimiter(per_ip=5, window_seconds=60, daily_total=100)
    limiter.check(FakeRequest())
    assert limiter.snapshot() == {
        "per_ip": 5,
        "window_seconds": 60,
        "daily_total": 100,
        "used_today": 1,
    }


def test_rejected_requests_do_not_consume_the_daily_budget():
    limiter = RateLimiter(per_ip=1, window_seconds=300, daily_total=100)
    request = FakeRequest()
    limiter.check(request)
    with pytest.raises(HTTPException):
        limiter.check(request)
    assert limiter.snapshot()["used_today"] == 1
