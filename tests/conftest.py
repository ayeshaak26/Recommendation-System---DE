"""Runs before any test module is imported: keep test output out of logs/pipeline.log."""

import tempfile
from pathlib import Path

from pipeline import config

config.LOG_FILE = Path(tempfile.mkdtemp(prefix="recsys-test-logs-")) / "pipeline.log"
