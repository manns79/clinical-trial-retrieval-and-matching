from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from typing import Any


def now_ms() -> float:
    return time.perf_counter() * 1000


def elapsed_ms(start_ms: float) -> float:
    return round(now_ms() - start_ms, 3)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    fields: Mapping[str, Any] | None = None,
) -> None:
    payload = {"event": event}
    if fields:
        payload.update(fields)
    logger.log(level, json.dumps(payload, sort_keys=True, default=str))
