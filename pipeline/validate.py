"""VALIDATE stage: data quality checks on the flattened movies table.

Two severities, so one bad field never costs us a whole movie unnecessarily:

  ERRORS  -> the row is quarantined (kept out of the staged layer, saved for review)
      - missing / non-positive movie_id        - duplicate movie_id
      - null or empty title                    - rating outside 0-10
      - missing or too-short overview          - unparseable release_date
      - adult title

  WARNINGS -> the field is repaired (set to null) and the row is kept
      - malformed poster_path  (the app falls back to the placeholder image)
      - malformed trailer key  (the app shows "no trailer available")

Nothing here raises on bad data: the pipeline keeps going and reports the counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

try:  # pandera >= 0.24 moved the pandas API here
    import pandera.pandas as pa
    from pandera.errors import SchemaErrors
except ImportError:  # older pandera
    import pandera as pa
    from pandera.errors import SchemaErrors

from pipeline import config
from pipeline.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class ValidationResult:
    """Outcome of validating one batch of movies."""

    valid: pd.DataFrame
    rejected: pd.DataFrame                       # same columns as the input + "reasons"
    warnings: dict[str, int] = field(default_factory=dict)

    @property
    def reason_counts(self) -> dict[str, int]:
        """How many rows failed each rule (a row can fail several)."""
        if self.rejected.empty:
            return {}
        return self.rejected["reasons"].str.split("; ").explode().value_counts().to_dict()


def build_schema() -> "pa.DataFrameSchema":
    """The hard rules. Each check's `error` text becomes the quarantine reason."""
    return pa.DataFrameSchema(
        {
            "movie_id": pa.Column(
                checks=pa.Check.gt(0, error="movie_id must be positive"), nullable=False
            ),
            "title": pa.Column(
                checks=pa.Check(lambda s: s.str.strip().str.len() > 0, error="empty title"),
                nullable=False,
            ),
            "overview": pa.Column(
                checks=pa.Check(
                    lambda s: s.str.strip().str.len() >= config.MIN_OVERVIEW_CHARS,
                    error="overview too short",
                ),
                nullable=False,
            ),
            "vote_average": pa.Column(
                checks=pa.Check.in_range(
                    config.RATING_MIN, config.RATING_MAX, error="rating outside 0-10"
                ),
                nullable=False,
            ),
            "release_date": pa.Column(
                checks=pa.Check(
                    lambda s: pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").notna(),
                    error="unparseable release_date",
                ),
                nullable=True,  # unreleased movies may have no date yet
            ),
            "adult": pa.Column(
                checks=pa.Check(lambda s: ~s.astype(bool), error="adult title"), nullable=True
            ),
        },
        strict=False,  # other columns pass through unchecked
    )


def repair_soft_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Null out malformed poster/trailer fields instead of rejecting the movie."""
    df = df.copy()
    warnings: dict[str, int] = {}
    rules = {
        "poster_path": (config.POSTER_PATH_REGEX, "malformed_poster_path"),
        "trailer_key": (config.YOUTUBE_KEY_REGEX, "malformed_trailer_key"),
    }
    for column, (pattern, label) in rules.items():
        present = df[column].notna()
        bad = present & ~df[column].astype("string").str.match(pattern, na=False)
        if bad.any():
            df.loc[bad, column] = None
        warnings[label] = int(bad.sum())
    # Informational: not errors, but worth tracking in the quality monitor.
    warnings["missing_poster"] = int(df["poster_path"].isna().sum())
    warnings["missing_trailer"] = int(df["trailer_key"].isna().sum())
    return df, warnings


def validate_movies(movies: pd.DataFrame) -> ValidationResult:
    """Split a flattened movies table into valid rows and quarantined rows."""
    df = movies.reset_index(drop=True)
    df, warnings = repair_soft_fields(df)
    reasons: dict[int, set[str]] = {}

    # Duplicates: keep the first occurrence, quarantine the rest.
    for idx in df.index[df["movie_id"].notna() & df.duplicated("movie_id", keep="first")]:
        reasons.setdefault(idx, set()).add("duplicate movie_id")

    # lazy=True collects every failure instead of stopping at the first one.
    try:
        build_schema().validate(df, lazy=True)
    except SchemaErrors as err:
        cases = err.failure_cases
        structural = cases[cases["index"].isna()]
        if not structural.empty:
            # A whole column is missing: that is a code bug, not bad data.
            raise ValueError(f"Schema problem, not a data problem:\n{structural}") from err
        for idx, column, check in zip(cases["index"], cases["column"], cases["check"]):
            reason = f"missing {column}" if check == "not_nullable" else str(check)
            reasons.setdefault(int(idx), set()).add(reason)

    bad_index = sorted(reasons)
    rejected = df.loc[bad_index].copy()
    rejected["reasons"] = ["; ".join(sorted(reasons[i])) for i in bad_index]
    valid = df.drop(index=bad_index).reset_index(drop=True)

    result = ValidationResult(valid=valid, rejected=rejected.reset_index(drop=True), warnings=warnings)
    log.info("Validation: %d valid, %d rejected %s", len(valid), len(rejected), result.reason_counts or "")
    return result
