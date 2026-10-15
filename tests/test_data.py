from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import polars as pl
import pytest

from cx_sentiment.data import (
    DataValidationError,
    audit_dataset,
    create_validation_sample,
    load_dataset,
)
from cx_sentiment.models import LABEL_TO_ID, SentimentLabel

LOCKED_SHA256 = "8fcf0f32af008e3f3620b41558faf9b63aa7d09408243e8435e43fab0df93f94"


def test_locked_test_is_balanced_valid_and_unchanged() -> None:
    path = Path("data/cx_sentiment_test_600.csv")
    before = path.read_bytes()
    frame = load_dataset(path, expected_split="test")
    audit = audit_dataset(frame)

    assert hashlib.sha256(before).hexdigest() == LOCKED_SHA256
    assert audit.n_rows == 600
    assert audit.languages == {code: 100 for code in ("de", "en", "es", "fr", "it", "pt")}
    assert audit.sentiments == {label.value: 200 for label in SentimentLabel}
    assert audit.normalized_duplicate_count == 0
    assert LABEL_TO_ID == {
        SentimentLabel.negative: 0,
        SentimentLabel.neutral: 1,
        SentimentLabel.positive: 2,
    }
    assert path.read_bytes() == before


def _write_source_archive(path: Path) -> None:
    fields = ["language", "text", "sentiment", "label"]
    split_rows: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "validation", "test"):
        rows = []
        for language in ("pt", "en", "es", "fr", "de", "it"):
            for sentiment, label in (("negative", "0"), ("neutral", "1"), ("positive", "2")):
                for i in range(40):
                    rows.append(
                        {
                            "language": language,
                            "text": f"{split} {language} {sentiment} unique example {i}",
                            "sentiment": sentiment,
                            "label": label,
                        }
                    )
        split_rows[split] = rows
    with zipfile.ZipFile(path, "w") as archive:
        for split, rows in split_rows.items():
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            archive.writestr(f"{split}.csv", buffer.getvalue())


def test_validation_sampler_is_deterministic_stratified_and_manifested(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    _write_source_archive(archive)
    out_a, out_b = tmp_path / "a", tmp_path / "b"

    first = create_validation_sample(archive, out_a, sample_size=600, seed=20261016)
    second = create_validation_sample(archive, out_b, sample_size=600, seed=20261016)

    assert (out_a / "validation_600.csv").read_bytes() == (
        out_b / "validation_600.csv"
    ).read_bytes()
    assert first.selected_source_indices == second.selected_source_indices
    assert first.frame.height == 600
    language_counts = first.frame["language"].value_counts().to_dicts()
    assert {row["language"]: row["count"] for row in language_counts} == {
        code: 100 for code in ("pt", "en", "es", "fr", "de", "it")
    }
    assert Counter(first.frame["sentiment"].to_list()) == {
        "negative": 200,
        "neutral": 200,
        "positive": 200,
    }
    manifest = json.loads((out_a / "validation_manifest.json").read_text())
    assert manifest["source_split"] == "validation"
    assert manifest["sampling_seed"] == 20261016
    assert len(manifest["selected_source_indices_zero_based"]) == 600
    assert manifest["source_archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_audit_rejects_normalized_cross_split_overlap(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    _write_source_archive(archive)
    records = load_dataset(Path("data/cx_sentiment_test_600.csv"), expected_split="test")
    duplicate = records.head(1).with_columns(
        pl.lit("other").alias("sample_id"), pl.lit("validation").alias("source_split")
    )

    with pytest.raises(DataValidationError, match="overlap"):
        audit_dataset(duplicate, comparison_frame=records)


def test_dataframe_schema_rejects_mismatched_label(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.csv"
    frame = load_dataset(Path("data/cx_sentiment_test_600.csv"), expected_split="test").head(1)
    frame.with_columns(pl.lit(2).alias("label")).write_csv(invalid)

    with pytest.raises(DataValidationError, match="label_matches_sentiment"):
        load_dataset(invalid, expected_split="test")
