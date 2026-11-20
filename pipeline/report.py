"""Execute a notebook with papermill and save a dated copy as a report.

    python -m pipeline.report                       # data quality report for today
    python -m pipeline.report --date 2026-11-20     # label the report with another date
    python -m pipeline.report --notebook 01_pipeline_runner -p STAGE load

Output: notebooks/reports/<notebook>_<YYYY-MM-DD>.ipynb (and .html with --html).

papermill runs the notebook top to bottom in a fresh kernel, replacing the values in the
cell tagged "parameters". The source notebook is never modified, so every scheduled run
leaves behind a self-contained, dated record of what the data looked like that day.
Re-running on the same day overwrites that day's report.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

from pipeline import config
from pipeline.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_NOTEBOOK = "02_data_quality_monitor"


def _coerce(value: str) -> Any:
    """Turn a command-line string into bool / None / int / float where it clearly is one."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def run_report(
    notebook: str = DEFAULT_NOTEBOOK,
    report_date: str | None = None,
    parameters: dict[str, Any] | None = None,
    html: bool = False,
) -> Path:
    """Execute a notebook and return the path of the dated output notebook.

    Args:
        notebook: file name in notebooks/, with or without ".ipynb".
        report_date: "YYYY-MM-DD" used in the output file name (default: today).
        parameters: values for the notebook's "parameters" cell.
        html: also export the executed notebook to HTML (code cells hidden).
    """
    import papermill as pm  # imported here so the rest of the pipeline works without it

    name = notebook.removesuffix(".ipynb")
    source = config.NOTEBOOK_DIR / f"{name}.ipynb"
    if not source.exists():
        raise FileNotFoundError(f"Notebook not found: {source}")

    report_date = report_date or date.today().isoformat()
    params = dict(parameters or {})
    if name == DEFAULT_NOTEBOOK:
        params.setdefault("REPORT_DATE", report_date)

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = config.REPORT_DIR / f"{name}_{report_date}.ipynb"
    log.info("Report started: %s -> %s", source.name, output)

    pm.execute_notebook(
        str(source), str(output),
        parameters=params,
        cwd=str(config.PROJECT_ROOT),
        kernel_name="python3",
        language="python",
        kernel_manager_class=_same_interpreter_kernel_manager(),
        progress_bar=False,
        log_output=False,
    )
    log.info("Report finished: %s", output)

    if html:
        from nbconvert import HTMLExporter

        body, _ = HTMLExporter(exclude_input=True).from_filename(str(output))
        output.with_suffix(".html").write_text(body, encoding="utf-8")
        log.info("HTML report written: %s", output.with_suffix(".html"))
    return output


def _same_interpreter_kernel_manager() -> type:
    """Kernel manager that starts the kernel with this interpreter (the project's venv).

    Without it, papermill uses whichever "python3" kernel is registered on the machine,
    which may be another Python installation that does not have the project's packages.
    Built lazily so importing this module does not require jupyter_client.
    """
    from jupyter_client.manager import AsyncKernelManager

    class SameInterpreterKernelManager(AsyncKernelManager):
        def format_kernel_cmd(self, extra_arguments: list[str] | None = None) -> list[str]:
            cmd = super().format_kernel_cmd(extra_arguments)
            return [sys.executable, *cmd[1:]]

    return SameInterpreterKernelManager


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Execute a notebook with papermill into a dated report.")
    parser.add_argument("--notebook", default=DEFAULT_NOTEBOOK, help=f"notebook name (default: {DEFAULT_NOTEBOOK})")
    parser.add_argument("--date", default=None, help="report date YYYY-MM-DD (default: today)")
    parser.add_argument("-p", "--param", nargs=2, action="append", default=[], metavar=("NAME", "VALUE"),
                        help="override a notebook parameter; repeatable")
    parser.add_argument("--html", action="store_true", help="also write an HTML copy")
    args = parser.parse_args(argv)

    try:
        output = run_report(args.notebook, args.date, {k: _coerce(v) for k, v in args.param}, args.html)
    except Exception:
        log.exception("Report failed")
        return 1
    print(f"Report written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
