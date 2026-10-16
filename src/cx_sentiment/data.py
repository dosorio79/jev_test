from __future__ import annotations

import hashlib
import io
import json
import random
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

import dataframely as dy
import polars as pl
from dataframely.exc import SchemaError, ValidationError

LANGUAGES = ("pt", "en", "es", "fr", "de", "it")
SENTIMENTS = ("negative", "neutral", "positive")
DATASET_COLUMNS = (
    "sample_id",
    "source_split",
    "source_row_index_zero_based",
    "language",
    "text",
    "sentiment",
    "label",
)
LOCKED_TEST_SHA256 = "8fcf0f32af008e3f3620b41558faf9b63aa7d09408243e8435e43fab0df93f94"


class DataValidationError(ValueError):
    """Raised when benchmark data violate the locked data contract."""


class SourceDatasetSchema(dy.Schema):
    language = dy.String()
    text = dy.String(min_length=1)
    sentiment = dy.String()
    label = dy.Int64(min=0, max=2)

    @dy.rule()
    def language_is_supported(cls) -> pl.Expr:
        return pl.col("language").is_in(LANGUAGES)

    @dy.rule()
    def sentiment_is_canonical(cls) -> pl.Expr:
        return pl.col("sentiment").is_in(SENTIMENTS)

    @dy.rule()
    def label_matches_sentiment(cls) -> pl.Expr:
        expected = pl.col("sentiment").replace_strict(
            {"negative": 0, "neutral": 1, "positive": 2}, return_dtype=pl.Int64
        )
        return pl.col("label") == expected


class BenchmarkDatasetSchema(SourceDatasetSchema):
    sample_id = dy.String(primary_key=True, min_length=1)
    source_split = dy.String()
    source_row_index_zero_based = dy.Int64(min=0, unique=True)

    @dy.rule()
    def source_split_is_known(cls) -> pl.Expr:
        return pl.col("source_split").is_in(["train", "validation", "test"])


@dataclass(frozen=True)
class DatasetAudit:
    n_rows: int
    languages: dict[str, int]
    sentiments: dict[str, int]
    normalized_duplicate_count: int
    cross_split_overlap_count: int


@dataclass(frozen=True)
class SampleResult:
    frame: pl.DataFrame
    selected_source_indices: list[int]


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _validate_frame(frame: pl.DataFrame, schema: type[dy.Schema], source: str) -> pl.DataFrame:
    try:
        schema.validate(frame)
    except (SchemaError, ValidationError) as exc:
        raise DataValidationError(f"invalid dataset {source}: {exc}") from exc
    return frame


def load_dataset(path: Path, *, expected_split: str | None = None) -> pl.DataFrame:
    try:
        frame = pl.read_csv(
            path,
            encoding="utf8-lossy",
            schema_overrides={"source_row_index_zero_based": pl.Int64, "label": pl.Int64},
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise DataValidationError(f"cannot read dataset {path}: {exc}") from exc
    if frame.columns != list(DATASET_COLUMNS):
        raise DataValidationError(
            f"invalid columns in {path}: expected {list(DATASET_COLUMNS)}, got {frame.columns}"
        )
    validated = _validate_frame(frame, BenchmarkDatasetSchema, str(path))
    if expected_split is not None:
        unexpected = validated.filter(pl.col("source_split") != expected_split)
        if not unexpected.is_empty():
            raise DataValidationError(
                f"dataset contains {unexpected.height} rows outside expected split "
                f"{expected_split!r}"
            )
    if expected_split == "test":
        _validate_locked_test(validated, path)
    return validated


def _validate_locked_test(frame: pl.DataFrame, path: Path) -> None:
    language_counts = dict(frame.group_by("language").len().iter_rows())
    sentiment_counts = dict(frame.group_by("sentiment").len().iter_rows())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if frame.height != 600 or language_counts != {code: 100 for code in LANGUAGES}:
        raise DataValidationError("locked test must contain 600 rows and 100 per language")
    if sentiment_counts != {label: 200 for label in SENTIMENTS}:
        raise DataValidationError("locked test must contain 200 rows per sentiment")
    if digest != LOCKED_TEST_SHA256:
        raise DataValidationError(
            f"locked test SHA256 mismatch: expected {LOCKED_TEST_SHA256}, got {digest}"
        )


def _normalized_texts(frame: pl.DataFrame) -> pl.Series:
    return frame["text"].map_elements(normalize_text, return_dtype=pl.String)


def audit_dataset(
    frame: pl.DataFrame,
    *,
    comparison_frame: pl.DataFrame | None = None,
) -> DatasetAudit:
    normalized = _normalized_texts(frame)
    duplicate_count = normalized.len() - normalized.n_unique()
    comparison = (
        set(_normalized_texts(comparison_frame).to_list())
        if comparison_frame is not None
        else set()
    )
    overlap_count = len(set(normalized.to_list()) & comparison)
    if duplicate_count:
        raise DataValidationError(f"found {duplicate_count} normalized duplicate records")
    if overlap_count:
        raise DataValidationError(f"found {overlap_count} normalized cross-split overlap records")
    language_counts = frame.group_by("language").len().sort("language")
    sentiment_counts = frame.group_by("sentiment").len().sort("sentiment")
    return DatasetAudit(
        n_rows=frame.height,
        languages=dict(language_counts.iter_rows()),
        sentiments=dict(sentiment_counts.iter_rows()),
        normalized_duplicate_count=duplicate_count,
        cross_split_overlap_count=overlap_count,
    )


def _read_archive_split(archive: zipfile.ZipFile, split: str) -> pl.DataFrame:
    try:
        raw = archive.read(f"{split}.csv")
    except KeyError as exc:
        raise DataValidationError(f"source archive is missing {split}.csv") from exc
    try:
        frame = pl.read_csv(
            io.BytesIO(raw), encoding="utf8-lossy", schema_overrides={"label": pl.Int64}
        )
    except pl.exceptions.PolarsError as exc:
        raise DataValidationError(f"cannot parse {split}.csv: {exc}") from exc
    expected_columns = ["language", "text", "sentiment", "label"]
    if set(frame.columns) != set(expected_columns):
        raise DataValidationError(
            f"invalid columns in {split}.csv: expected {expected_columns}, got {frame.columns}"
        )
    frame = frame.select(expected_columns)
    return _validate_frame(frame, SourceDatasetSchema, f"{split}.csv")


def create_validation_sample(
    archive_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 600,
    seed: int = 20261016,
) -> SampleResult:
    if sample_size != 600:
        raise DataValidationError("v1 validation sample size must be exactly 600")
    if not archive_path.is_file():
        raise DataValidationError(
            f"source archive not found: {archive_path}; pass --archive "
            "data/raw/financial_sentiment_hf_v2.zip"
        )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            splits = {
                split: _read_archive_split(archive, split)
                for split in ("train", "validation", "test")
            }
    except zipfile.BadZipFile as exc:
        raise DataValidationError(f"invalid ZIP archive: {archive_path}") from exc

    validation = splits["validation"].with_row_index("source_row_index_zero_based")
    rng = random.Random(seed)
    selected_indices: list[int] = []
    strata: dict[str, int] = {}
    for language_index, language in enumerate(LANGUAGES):
        for sentiment_index, sentiment in enumerate(SENTIMENTS):
            target = 34 if sentiment_index == language_index % 3 else 33
            candidates = validation.filter(
                (pl.col("language") == language) & (pl.col("sentiment") == sentiment)
            )["source_row_index_zero_based"].to_list()
            if len(candidates) < target:
                raise DataValidationError(
                    f"validation stratum {language}/{sentiment} has {len(candidates)} rows; "
                    f"needs {target}"
                )
            selected_indices.extend(rng.sample(candidates, target))
            strata[f"{language}/{sentiment}"] = target
    rng.shuffle(selected_indices)
    order = pl.DataFrame(
        {
            "source_row_index_zero_based": selected_indices,
            "_sample_order": range(len(selected_indices)),
        }
    )
    selected = (
        order.join(validation, on="source_row_index_zero_based", how="left")
        .sort("_sample_order")
        .drop("_sample_order")
        .with_columns(
            pl.struct(["source_row_index_zero_based", "text"])
            .map_elements(
                lambda row: (
                    "cxv_"
                    + hashlib.sha256(
                        f"validation:{row['source_row_index_zero_based']}:{row['text']}".encode()
                    ).hexdigest()[:16]
                ),
                return_dtype=pl.String,
            )
            .alias("sample_id"),
            pl.lit("validation").alias("source_split"),
        )
        .select(DATASET_COLUMNS)
    )
    selected = _validate_frame(selected, BenchmarkDatasetSchema, "sampled validation")
    other = pl.concat([splits["train"], splits["test"]], how="vertical")
    audit = audit_dataset(selected, comparison_frame=other)

    output_dir.mkdir(parents=True, exist_ok=True)
    selected.write_csv(output_dir / "validation_600.csv")
    manifest = {
        "source_archive": archive_path.name,
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "source_split": "validation",
        "sampling_seed": seed,
        "n_rows": selected.height,
        "languages": audit.languages,
        "sentiments": audit.sentiments,
        "strata": strata,
        "selected_source_indices_zero_based": selected_indices,
        "normalized_cross_split_overlap_count": audit.cross_split_overlap_count,
        "known_limitation": (
            "Synthetic template variations may be semantic near-duplicates; exact normalization "
            "does not establish independent real-world generalization."
        ),
    }
    (output_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return SampleResult(frame=selected, selected_source_indices=selected_indices)
