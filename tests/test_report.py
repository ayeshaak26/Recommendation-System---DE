"""Report tests. One test really executes a tiny notebook through papermill (a few seconds)."""

from __future__ import annotations

import json

import pytest

from pipeline import config, report

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("papermill")


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NOTEBOOK_DIR", tmp_path / "notebooks")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "notebooks" / "reports")
    config.NOTEBOOK_DIR.mkdir()


def write_notebook(name: str, body: str) -> None:
    nb = nbformat.v4.new_notebook()
    params = nbformat.v4.new_code_cell("GREETING = 'hello'\nTIMES = 1")
    params.metadata["tags"] = ["parameters"]
    nb.cells = [params, nbformat.v4.new_code_cell(body)]
    nbformat.write(nb, config.NOTEBOOK_DIR / f"{name}.ipynb")


def test_coerce_turns_cli_strings_into_python_values():
    assert [report._coerce(v) for v in ("true", "False", "none", "7", "0.5", "load")] == [
        True, False, None, 7, 0.5, "load"
    ]


def test_missing_notebook_gives_a_clear_error_and_exit_code_1():
    with pytest.raises(FileNotFoundError, match="Notebook not found"):
        report.run_report("does_not_exist")
    assert report.main(["--notebook", "does_not_exist"]) == 1


def test_notebook_is_executed_into_a_dated_report_with_parameters_applied():
    write_notebook("demo", "import sys\nprint(GREETING * TIMES)\nprint(sys.executable)")

    output = report.run_report("demo.ipynb", report_date="2026-11-20", parameters={"GREETING": "hi", "TIMES": 2})

    assert output == config.REPORT_DIR / "demo_2026-11-20.ipynb"
    printed = "".join(
        "".join(o.get("text", "")) for c in json.loads(output.read_text("utf-8"))["cells"] for o in c.get("outputs", [])
    )
    assert "hihi" in printed
    import sys
    assert sys.executable in printed  # ran in this interpreter, not some other registered kernel
    source = json.loads((config.NOTEBOOK_DIR / "demo.ipynb").read_text("utf-8"))
    assert all(not c.get("outputs") for c in source["cells"])  # the source notebook is untouched


def test_failing_notebook_returns_exit_code_1():
    write_notebook("broken", "raise ValueError('boom')")

    assert report.main(["--notebook", "broken"]) == 1
