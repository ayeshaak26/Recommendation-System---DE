"""Orchestrator tests. Extract and the model refresh are replaced by fakes, so no network
or PyTorch is needed; validate/transform/load run for real against a temporary database."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

from pipeline import config, extract, refresh_model, run_pipeline
from test_transform import raw_movie

TODAY = "2026-11-20"


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    for name in ("RAW_DIR", "STAGED_DIR", "QUARANTINE_DIR", "CURATED_DIR", "LOG_DIR", "REPORT_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "curated_dir" / "movies.db")


@pytest.fixture
def calls(monkeypatch):
    """Fake extract (writes the given raw movies) and fake refresh; records how they were called."""
    seen = SimpleNamespace(extract=[], refresh=[], movies=[raw_movie(1), raw_movie(2, title="Tenet")])

    def fake_extract(full_refresh=False, **_):
        seen.extract.append(full_refresh)
        folder = config.RAW_DIR / TODAY / "movies"
        folder.mkdir(parents=True, exist_ok=True)
        for movie in seen.movies:
            (folder / f"{movie['id']}.json").write_text(json.dumps(movie), encoding="utf-8")
        return SimpleNamespace(run_date=TODAY, rows_extracted=len(seen.movies), failed=0)

    def fake_refresh(force=False, **_):
        seen.refresh.append(force)
        return SimpleNamespace(refreshed=True, reason="2 new and 0 removed movies", winner="tfidf")

    monkeypatch.setattr(extract, "run_extract", fake_extract)
    monkeypatch.setattr(refresh_model, "run_refresh", fake_refresh)
    return seen


def runs() -> list[tuple]:
    with closing(sqlite3.connect(config.DB_PATH)) as conn:
        return conn.execute(
            "SELECT stage, full_refresh, status, rows_extracted, rows_rejected, rows_loaded, "
            "rows_inserted, rows_updated, model_refreshed FROM pipeline_runs ORDER BY run_id"
        ).fetchall()


def test_all_runs_every_stage_and_records_one_row(calls):
    calls.movies.append(raw_movie(3, title=""))  # will be quarantined
    summary = run_pipeline.run("all")

    assert runs() == [("all", 0, "success", 3, 1, 2, 2, 0, 1)]
    assert summary["rows_inserted"] == 2 and summary["model"].startswith("rebuilt")
    assert (calls.extract, calls.refresh) == ([False], [False])


def test_running_all_twice_is_idempotent(calls):
    run_pipeline.run("all")
    run_pipeline.run("all")

    assert [(r[6], r[7]) for r in runs()] == [(2, 0), (0, 2)]  # second run: 0 inserted, 2 updated


def test_full_refresh_is_passed_to_extract_and_forces_the_model(calls):
    run_pipeline.run("all", full_refresh=True)

    assert (calls.extract, calls.refresh) == ([True], [True])
    assert runs()[0][1] == 1


def test_all_skips_transform_and_load_when_nothing_was_extracted(calls):
    calls.movies.clear()
    summary = run_pipeline.run("all")

    assert runs() == [("all", 0, "success", 0, 0, 0, 0, 0, 1)]
    assert not (config.STAGED_DIR / TODAY).exists()
    assert calls.refresh == [False] and "rows_loaded" not in summary


def test_single_stages_run_only_themselves(calls):
    run_pipeline.run("extract")
    assert not (config.STAGED_DIR / TODAY).exists() and calls.refresh == []

    run_pipeline.run("transform")
    assert (config.STAGED_DIR / TODAY / "movies.parquet").exists()

    run_pipeline.run("load", run_date=TODAY)
    assert [(r[0], r[5]) for r in runs()] == [("extract", 0), ("transform", 0), ("load", 2)]
    assert calls.extract == [False] and calls.refresh == []


def test_failure_is_recorded_and_gives_exit_code_1(calls, monkeypatch):
    def boom(**_):
        raise RuntimeError("TMDB is down")

    monkeypatch.setattr(extract, "run_extract", boom)

    assert run_pipeline.main(["--stage", "all"]) == 1
    assert [r[2] for r in runs()] == ["failed"]


def test_cli_success_returns_0_and_prints_a_summary(calls, capsys):
    assert run_pipeline.main(["--stage", "all", "--full-refresh"]) == 0
    assert "rows_inserted" in capsys.readouterr().out


def test_cli_rejects_an_unknown_stage(calls):
    with pytest.raises(SystemExit):
        run_pipeline.main(["--stage", "magic"])
