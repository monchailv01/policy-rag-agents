"""Request throttling for the public deployment.

The chat endpoint is the only expensive one — every turn spends real tokens on
the operator's API key — so it is guarded by two independent limits:

* a per-client sliding window, which stops one visitor from monopolising it;
* a global daily budget, which caps the worst case for the whole service no
  matter how many clients show up.

State is in-process and unsynchronised on purpose: FastAPI serves this app on a
single event loop and the check below contains no ``await``, so it runs
atomically without a lock. A multi-worker deployment would need Redis instead.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import date

from fastapi import HTTPException, Request

#: Headers a Cloudflare Tunnel sets, in the order we trust them.
_FORWARDED_HEADERS = ("cf-connecting-ip", "x-forwarded-for", "x-real-ip")


def client_ip(request: Request) -> str:
    """Best available client address, looking through the Cloudflare edge."""
    for header in _FORWARDED_HEADERS:
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding window per client, plus a hard daily total."""

    def __init__(self, *, per_ip: int, window_seconds: int, daily_total: int) -> None:
        self.per_ip = per_ip
        self.window_seconds = window_seconds
        self.daily_total = daily_total
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._day = date.today()
        self._today_total = 0

    def _roll_day(self) -> None:
        today = date.today()
        if today != self._day:
            self._day, self._today_total = today, 0
            self._hits.clear()

    def check(self, request: Request) -> None:
        """Record one request, or raise ``429`` if a limit is already reached."""
        self._roll_day()

        if self.daily_total and self._today_total >= self.daily_total:
            raise HTTPException(
                status_code=429,
                detail=(
                    "This demo has reached its daily request budget. "
                    "Please try again tomorrow."
                ),
            )

        now = time.monotonic()
        window = self._hits[client_ip(request)]
        while window and now - window[0] > self.window_seconds:
            window.popleft()

        if self.per_ip and len(window) >= self.per_ip:
            retry_after = int(self.window_seconds - (now - window[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Too many requests. You may send {self.per_ip} "
                    f"question{'' if self.per_ip == 1 else 's'} every "
                    f"{self.window_seconds // 60} minutes."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)
        self._today_total += 1

    def snapshot(self) -> dict:
        """Current usage, surfaced by ``/api/meta`` for the UI footer."""
        self._roll_day()
        return {
            "per_ip": self.per_ip,
            "window_seconds": self.window_seconds,
            "daily_total": self.daily_total,
            "used_today": self._today_total,
        }
