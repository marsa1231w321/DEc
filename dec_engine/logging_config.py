from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import TextIO


_LOGGER_NAME = "dec_engine"
_MANAGED_HANDLERS: list[logging.Handler] = []


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "context": getattr(record, "dec_context", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)


class StructuredLogger(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger, context: dict):
        super().__init__(logger, {})
        self.context = _validate_context(context)

    def bind(self, **context) -> StructuredLogger:
        return StructuredLogger(
            self.logger, {**self.context, **_validate_context(context)}
        )

    def process(self, message, kwargs):
        existing_extra = kwargs.get("extra")
        call_extra = dict(existing_extra) if existing_extra is not None else {}
        call_context = _validate_context(call_extra.pop("dec_context", {}))
        kwargs["extra"] = {
            **call_extra,
            "dec_context": {**self.context, **call_context},
        }
        return message, kwargs


def _validate_context(context: dict) -> dict:
    if not isinstance(context, dict):
        raise TypeError("Log context must be a dictionary.")
    for key, value in context.items():
        if not isinstance(key, str) or not isinstance(
            value, (str, int, float, bool, type(None))
        ):
            raise TypeError("Log context must contain scalar values.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Log context cannot contain non-finite numbers.")
    return context


def _level_value(level: str | int) -> int:
    if isinstance(level, bool):
        raise ValueError(f"Invalid logging level: {level!r}")
    if isinstance(level, int):
        if level in {
            logging.NOTSET,
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        }:
            return level
    elif isinstance(level, str):
        value = logging.getLevelName(level.upper())
        if isinstance(value, int):
            return value
    raise ValueError(f"Invalid logging level: {level!r}")


def configure_logging(level: str | int | None = None, *, stream: TextIO | None = None):
    """Configure structured developer logs without exposing them in editor UI."""
    configured_level = (
        level if level is not None else os.getenv("DEC_LOG_LEVEL", "INFO")
    )
    numeric_level = _level_value(configured_level)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(numeric_level)
    logger.propagate = False
    for handler in _MANAGED_HANDLERS:
        logger.removeHandler(handler)
    _MANAGED_HANDLERS.clear()

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(numeric_level)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    _MANAGED_HANDLERS.append(handler)


def get_logger(name: str, **context) -> StructuredLogger:
    if not name or not isinstance(name, str):
        raise ValueError("Logger name must be a non-empty string.")
    logger_name = (
        name
        if name == _LOGGER_NAME or name.startswith(f"{_LOGGER_NAME}.")
        else f"{_LOGGER_NAME}.{name}"
    )
    return StructuredLogger(logging.getLogger(logger_name), context)
