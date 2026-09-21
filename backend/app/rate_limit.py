from collections import defaultdict, deque
from time import monotonic
from fastapi import HTTPException, Request, status
from .config import get_settings

_visits: dict[str, deque[float]] = defaultdict(deque)


def enforce_public_rate_limit(request: Request) -> None:
    """Process-local limiter; use Redis-backed middleware before multi-instance deployment."""
    client = request.client.host if request.client else "unknown"
    now = monotonic()
    bucket = _visits[client]
    while bucket and bucket[0] <= now - 60:
        bucket.popleft()
    if len(bucket) >= get_settings().public_rate_limit_per_minute:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests; retry shortly.")
    bucket.append(now)
