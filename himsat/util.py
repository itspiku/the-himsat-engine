from __future__ import annotations

import json
import logging
import random
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")
log = logging.getLogger("himsat")


def retry(fn: Callable[..., T], attempts: int = 4, base_delay: float = 1.5,
          exceptions: tuple[type[BaseException], ...] = (Exception,)) -> Callable[..., T]:
    """Wrap ``fn`` with exponential backoff + jitter."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except exceptions as e:  # noqa: PERF203
                if i == attempts - 1:
                    raise
                delay = base_delay * (2**i) * (0.75 + random.random() / 2)
                log.warning("retry %d/%d after %s: %s", i + 1, attempts - 1, type(e).__name__, e)
                time.sleep(delay)
        raise RuntimeError("unreachable")

    return wrapper


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": datetime.fromtimestamp(record.created, UTC).isoformat(), "level": record.levelname,
             "logger": record.name, "msg": record.getMessage()}
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, ensure_ascii=False)


def setup_logging(level: str = "INFO", as_json: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    if as_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("rasterio", "urllib3", "httpx", "httpcore", "botocore", "azure", "pystac_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def parse_date(s: str) -> datetime:
    return utc(datetime.fromisoformat(s))
