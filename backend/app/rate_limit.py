from collections import defaultdict, deque
from functools import lru_cache
import logging
from time import monotonic
from fastapi import HTTPException, Request, status
from .config import get_settings

_visits: dict[str, deque[float]] = defaultdict(deque)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _redis_client():
    """Lazily connect so local tests and degraded dev mode still work."""
    import redis
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True, socket_timeout=0.2)


def _client_identity(request: Request) -> str:
    # Do not trust X-Forwarded-For by default: clients can forge it unless a
    # gateway strips/sets it. Configure that policy at the reverse proxy.
    return request.client.host if request.client else "unknown"


def _raise_limited() -> None:
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Quá nhiều yêu cầu; vui lòng thử lại sau ít phút.",
        headers={"Retry-After": "60"},
    )


def _enforce_local(key: str, limit: int) -> None:
    now = monotonic()
    bucket = _visits[key]
    while bucket and bucket[0] <= now - 60:
        bucket.popleft()
    if len(bucket) >= limit:
        _raise_limited()
    bucket.append(now)


def _enforce_rate_limit(request: Request, *, scope: str, limit: int) -> None:
    """Distributed fixed-window limiter with process-local safe fallback."""
    key = f"lawrag:ratelimit:{scope}:{_client_identity(request)}"
    settings = get_settings()
    if settings.rate_limit_redis_enabled:
        try:
            client = _redis_client()
            count = int(client.incr(key))
            if count == 1:
                client.expire(key, 60)
            if count > limit:
                _raise_limited()
            return
        except HTTPException:
            raise
        except Exception as exc:
            # Availability is preferable to making a legal lookup impossible;
            # local fallback still protects each API instance and leaves an
            # observable warning for operations.
            logger.warning("Redis rate limiter unavailable; using local fallback: %s", exc)
    _enforce_local(key, limit)


def enforce_public_rate_limit(request: Request) -> None:
    _enforce_rate_limit(request, scope="chat", limit=get_settings().public_rate_limit_per_minute)


def enforce_admin_login_rate_limit(request: Request) -> None:
    _enforce_rate_limit(request, scope="admin-login", limit=get_settings().admin_login_rate_limit_per_minute)
