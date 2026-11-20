"""Pipeline orchestrator: one command that runs the stages in order and records the run.

    python run_pipeline.py --stage all              # extract -> validate/transform -> load -> refresh model
    python run_pipeline.py --stage all --full-refresh
    python run_pipeline.py --stage extract          # one stage on its own
    python run_pipeline.py --stage transform --run-date 2026-11-20

Design notes
------------
* Every run - whatever the stage - gets one row in `pipeline_runs` (via load.track_run),
  including failed runs, so the monitoring notebook can chart them.
* The process exit code is 0 on success and 1 on failure. Task Scheduler shows that
  code as "Last Run Result", so a broken run is visible without opening a log.
* Each stage is idempotent, so the fix for any failure is simply "run it again".
* "validate" has no flag of its own: validation is the first half of the transform
  stage, so bad rows can never reach the staged layer.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any

from pipeline import config, extract, load, refresh_model, transform
from pipeline.logging_setup import get_logger

log = get_logger(__name__)

STAGES = ("all", "extract", "transform", "load", "refresh")


def _has_raw_movies(run_date: str) -> bool:
    """True if the raw layer holds at least one movie file for this date."""
    folder = config.RAW_DIR / run_date / "movies"
    return folder.is_dir() and any(folder.glob("*.json"))


def run(stage: str = "all", full_refresh: bool = False, run_date: str | None = None) -> dict[str, Any]:
    """Run one stage, or all of them, inside a tracked pipeline run.

    Args:
        stage: one of STAGES.
        full_refresh: re-fetch every movie and force a model rebuild.
        run_date: "YYYY-MM-DD" folder for transform/load when run on their own
            (default: the latest). Ignored by "all", which always processes today's extract.

    Returns:
        The metrics written to pipeline_runs, plus a short "model" note.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose from {STAGES}")

    config.ensure_directories()
    log.info("=== Pipeline run started | stage=%s full_refresh=%s ===", stage, full_refresh)
    summary: dict[str, Any] = {}

    with load.track_run(stage, full_refresh) as metrics:
        # ------------------------------ EXTRACT ------------------------------ #
        if stage in ("all", "extract"):
            extracted = extract.run_extract(full_refresh=full_refresh)
            metrics["rows_extracted"] = extracted.rows_extracted
            if extracted.failed:
                log.warning("%d movies could not be fetched; the next run will retry them", extracted.failed)
            if stage == "all":
                run_date = extracted.run_date

        # In "all" mode, a day with no new or changed movies has nothing to transform or load.
        has_data = stage != "all" or _has_raw_movies(run_date or date.today().isoformat())
        if not has_data:
            log.info("No new or changed movies today - skipping transform and load.")

        # ------------------------ VALIDATE + TRANSFORM ----------------------- #
        if stage in ("all", "transform") and has_data:
            transformed = transform.run_transform(run_date)
            metrics["rows_rejected"] = transformed.rows_rejected
            run_date = transformed.run_date

        # -------------------------------- LOAD ------------------------------- #
        if stage in ("all", "load") and has_data:
            loaded = load.run_load(run_date)
            metrics.update(
                rows_loaded=loaded.rows_loaded,
                rows_inserted=loaded.rows_inserted,
                rows_updated=loaded.rows_updated,
            )

        # ------------------------------ REFRESH ------------------------------ #
        if stage in ("all", "refresh"):
            refreshed = refresh_model.run_refresh(force=full_refresh)
            metrics["model_refreshed"] = refreshed.refreshed
            summary["model"] = (
                f"rebuilt ({refreshed.reason}); winner: {refreshed.winner}"
                if refreshed.refreshed else f"unchanged ({refreshed.reason})"
            )

        summary.update(metrics)

    log.info("=== Pipeline run finished | %s ===", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Run the TMDB movie pipeline.")
    parser.add_argument("--stage", choices=STAGES, default="all", help="stage to run (default: all)")
    parser.add_argument("--full-refresh", action="store_true",
                        help="re-fetch every movie and force a model rebuild")
    parser.add_argument("--run-date", default=None,
                        help="YYYY-MM-DD folder for --stage transform/load (default: latest)")
    args = parser.parse_args(argv)

    try:
        summary = run(args.stage, args.full_refresh, args.run_date)
    except KeyboardInterrupt:
        log.error("Pipeline interrupted by the user.")
        return 1
    except Exception:
        # The full traceback goes to logs/pipeline.log; the failure is also in pipeline_runs.
        log.exception("Pipeline run failed")
        return 1

    print("Pipeline run succeeded:")
    for key, value in summary.items():
        print(f"  {key:<16} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
