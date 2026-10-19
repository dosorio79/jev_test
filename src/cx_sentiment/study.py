from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict

from cx_sentiment.comparison import SavedRunInput, dataset_fingerprint
from cx_sentiment.data import load_dataset
from cx_sentiment.metrics import evaluate_predictions
from cx_sentiment.models import Prediction, SentimentLabel
from cx_sentiment.routing import HybridResult, hybrid_prediction
from cx_sentiment.study_config import StudyConfig


class LanguageThresholdPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    language: str
    support: int
    macro_f1: float
    openai_macro_f1: float
    jev_coverage: float


class ConditionalErrors(BaseModel):
    model_config = ConfigDict(frozen=True)

    accepted_error_rate: float | None
    deferred_error_rate: float | None


class ThresholdPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    threshold: float
    accuracy: float
    macro_f1: float
    macro_f1_delta_from_openai: float
    macro_f1_delta_ci95: tuple[float, float] | None
    fallback_fraction: float
    jev_coverage: float
    accepted_error_rate: float | None
    deferred_error_rate: float | None
    cost_usd_per_1000: float | None
    by_language: tuple[LanguageThresholdPoint, ...]
    qualifies_quality: bool
    qualifies_languages: bool
    non_dominated: bool = False


class OfflineStudyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    measurement_scope: str = "simulated"
    dataset_sha256: str
    support: int
    openai_macro_f1: float
    thresholds: tuple[ThresholdPoint, ...]
    candidate_thresholds: tuple[float, ...]
    recommendation_status: str


def _read_saved_run(
    directory: Path, expected_provider: str
) -> tuple[pl.DataFrame, list[Prediction]]:
    try:
        experiment = SavedRunInput.model_validate(
            yaml.safe_load((directory / "experiment.yaml").read_text(encoding="utf-8"))
        )
        input_path = (
            experiment.input
            if experiment.input.is_absolute()
            else directory / experiment.input
        )
        frame = load_dataset(input_path).head(experiment.limit).sort("sample_id")
        report = json.loads((directory / "report_metadata.json").read_text(encoding="utf-8"))
        metadata = report["metadata"]
        predictions = [
            Prediction.model_validate_json(line)
            for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read saved {expected_provider} run: {directory}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid saved metadata: {directory}")
    execution_path = directory / "execution.json"
    if metadata.get("complete") is False:
        raise ValueError(f"incomplete run cannot be analyzed: {directory}")
    if execution_path.exists():
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if not isinstance(execution, dict) or execution.get("complete") is not True:
            raise ValueError(f"incomplete run cannot be analyzed: {directory}")
    if frame.is_empty() or set(frame["source_split"].unique().to_list()) != {"validation"}:
        raise ValueError("offline threshold analysis requires only validation records")
    ids = [prediction.sample_id for prediction in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(frame["sample_id"].to_list()):
        raise ValueError(f"prediction IDs do not match saved source records: {directory}")
    identities = {(item.provider, item.model_id, item.prompt_version) for item in predictions}
    if len(identities) != 1 or next(iter(identities))[0] != expected_provider:
        raise ValueError(f"incompatible provider, model, or prompt metadata: {directory}")
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    return frame, [by_id[sample_id] for sample_id in frame["sample_id"].to_list()]


def _paired_delta_interval(
    truth: list[str],
    languages: list[str],
    hybrid: list[Prediction],
    fallback: list[Prediction],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float] | None:
    if samples <= 0 or len(truth) < 2:
        return None
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(truth)) for _ in truth]
        sample_truth = [truth[index] for index in indices]
        sample_languages = [languages[index] for index in indices]
        hybrid_score = evaluate_predictions(
            sample_truth,
            [hybrid[index] for index in indices],
            languages=sample_languages,
            bootstrap_samples=0,
        ).macro_f1
        fallback_score = evaluate_predictions(
            sample_truth,
            [fallback[index] for index in indices],
            languages=sample_languages,
            bootstrap_samples=0,
        ).macro_f1
        deltas.append(hybrid_score - fallback_score)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _error_rate(
    indices: list[int], truth: list[str], predictions: list[Prediction]
) -> float | None:
    if not indices:
        return None
    errors = sum(
        predictions[index].predicted_label is None
        or predictions[index].predicted_label != SentimentLabel(truth[index])
        for index in indices
    )
    return errors / len(indices)


def _mark_non_dominated(points: list[ThresholdPoint]) -> list[ThresholdPoint]:
    eligible = [
        point
        for point in points
        if point.qualifies_quality
        and point.qualifies_languages
        and point.cost_usd_per_1000 is not None
    ]
    frontier: set[float] = set()
    for candidate in eligible:
        dominated = any(
            other.threshold != candidate.threshold
            and other.macro_f1 >= candidate.macro_f1
            and other.cost_usd_per_1000 <= candidate.cost_usd_per_1000  # type: ignore[operator]
            and (
                other.macro_f1 > candidate.macro_f1
                or other.cost_usd_per_1000 < candidate.cost_usd_per_1000  # type: ignore[operator]
            )
            for other in eligible
        )
        if not dominated:
            frontier.add(candidate.threshold)
    return [
        point.model_copy(update={"non_dominated": point.threshold in frontier})
        for point in points
    ]


def analyze_saved_validation(config: StudyConfig) -> OfflineStudyResult:
    jev_frame, jev = _read_saved_run(config.inputs.jev_run, "jev")
    openai_frame, openai = _read_saved_run(config.inputs.openai_run, "openai")
    fingerprint = dataset_fingerprint(jev_frame)
    if fingerprint != dataset_fingerprint(openai_frame):
        raise ValueError("saved runs do not contain identical source records")
    if any(item.predicted_label is not None and item.scores_by_class is None for item in jev):
        raise ValueError("successful Jev predictions require complete class probability vectors")

    truth = jev_frame["sentiment"].to_list()
    languages = jev_frame["language"].to_list()
    openai_metrics = evaluate_predictions(truth, openai, languages=languages, bootstrap_samples=0)
    points: list[ThresholdPoint] = []
    for threshold_point in config.thresholds:
        threshold = threshold_point / 100
        routed: list[HybridResult] = [
            hybrid_prediction(decision, fallback, threshold=threshold)
            for decision, fallback in zip(jev, openai, strict=True)
        ]
        hybrid = [item.prediction for item in routed]
        metrics = evaluate_predictions(truth, hybrid, languages=languages, bootstrap_samples=0)
        accepted = [index for index, item in enumerate(routed) if item.route == "decision"]
        deferred = [index for index, item in enumerate(routed) if item.route != "decision"]
        costs = [item.effective_cost_usd for item in routed]
        language_points: list[LanguageThresholdPoint] = []
        for language in sorted(set(languages)):
            indices = [index for index, value in enumerate(languages) if value == language]
            language_metrics = evaluate_predictions(
                [truth[index] for index in indices],
                [hybrid[index] for index in indices],
                languages=[language] * len(indices),
                bootstrap_samples=0,
            )
            language_openai = evaluate_predictions(
                [truth[index] for index in indices],
                [openai[index] for index in indices],
                languages=[language] * len(indices),
                bootstrap_samples=0,
            )
            language_points.append(
                LanguageThresholdPoint(
                    language=language,
                    support=len(indices),
                    macro_f1=language_metrics.macro_f1,
                    openai_macro_f1=language_openai.macro_f1,
                    jev_coverage=sum(index in accepted for index in indices) / len(indices),
                )
            )
        points.append(
            ThresholdPoint(
                threshold=threshold,
                accuracy=metrics.accuracy,
                macro_f1=metrics.macro_f1,
                macro_f1_delta_from_openai=metrics.macro_f1 - openai_metrics.macro_f1,
                macro_f1_delta_ci95=_paired_delta_interval(
                    truth,
                    languages,
                    hybrid,
                    openai,
                    samples=config.bootstrap_samples,
                    seed=config.seed + threshold_point,
                ),
                fallback_fraction=len(deferred) / len(routed),
                jev_coverage=len(accepted) / len(routed),
                accepted_error_rate=_error_rate(accepted, truth, jev),
                deferred_error_rate=_error_rate(deferred, truth, jev),
                cost_usd_per_1000=(
                    sum(cost for cost in costs if cost is not None) * 1000 / len(costs)
                    if all(cost is not None for cost in costs)
                    else None
                ),
                by_language=tuple(language_points),
                qualifies_quality=(
                    metrics.macro_f1
                    >= openai_metrics.macro_f1 - config.max_macro_f1_delta
                ),
                qualifies_languages=all(
                    item.macro_f1
                    >= item.openai_macro_f1 - config.max_language_f1_delta
                    for item in language_points
                ),
            )
        )
    points = _mark_non_dominated(points)
    candidates = tuple(point.threshold for point in points if point.non_dominated)
    return OfflineStudyResult(
        dataset_sha256=fingerprint,
        support=len(truth),
        openai_macro_f1=openai_metrics.macro_f1,
        thresholds=tuple(points),
        candidate_thresholds=candidates,
        recommendation_status=(
            "candidate thresholds available from validation simulation"
            if candidates
            else "no recommendation because constraints or required costs are unavailable"
        ),
    )
