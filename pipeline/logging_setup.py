"""Shared logging: console output plus a rotating file in logs/pipeline.log."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from pipeline import config

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the shared "pipeline" root, configuring it on first use.

    Handlers are attached once to the "pipeline" root logger, so calling this
    from every module (or re-running a notebook cell) never duplicates log lines.
    """
    root = logging.getLogger("pipeline")
    if not root.handlers:
        config.ensure_directories()
        root.setLevel(config.LOG_LEVEL)
        formatter = logging.Formatter(_FORMAT)

        file_handler = RotatingFileHandler(
            config.LOG_FILE,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
        root.propagate = False

    short_name = name.removeprefix("pipeline.")
    return root if short_name in ("pipeline", "__main__") else root.getChild(short_name)
