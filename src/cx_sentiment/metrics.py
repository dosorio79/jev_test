from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import accuracy_score, confusion_matrix

from cx_sentiment.models import Prediction, SentimentLabel

LABELS = [label.value for label in SentimentLabel]
UNRESOLVED = "__unresolved__"


class EvaluationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    support: int
    accuracy: float
    macro_f1: float
    operational_coverage: float
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float | int]]
    by_language: dict[str, dict[str, float | int]]
    accuracy_ci95: tuple[float, float] | None
    macro_f1_ci95: tuple[float, float] | None
    ece: float | None
    brier_score: float | None
    latency_p50_ms: float
    latency_p95_ms: float
    latency_by_provider: dict[str, dict[str, float | int]]
    usage_input_tokens_total: int
    usage_output_tokens_total: int
    usage_records_fraction: float
    reported_credits_total: float | None
    cost_usd_known_total: float
    cost_usd_total: float | None
    cost_usd_per_1000: float | None
    cost_priced_fraction: float


def _point_metrics(
    truth: Sequence[str], predicted: Sequence[str]
) -> tuple[float, float, list[list[int]], dict[str, dict[str, float | int]]]:
    accuracy = float(accuracy_score(truth, predicted))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        pairs = list(zip(truth, predicted, strict=True))
        true_positive = sum(t == label and p == label for t, p in pairs)
        false_positive = sum(t != label and p == label for t, p in pairs)
        false_negative = sum(t == label and p != label for t, p in pairs)
        support = sum(value == label for value in truth)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    macro_f1 = float(np.mean(f1_values))
    matrix = confusion_matrix(truth, predicted, labels=LABELS).astype(int).tolist()
    return accuracy, macro_f1, matrix, per_class


def _bootstrap(
    truth: list[str], predicted: list[str], samples: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if samples <= 0 or len(truth) < 2:
        return None
    rng = random.Random(20261015)
    accuracies: list[float] = []
    macro_scores: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(truth)) for _ in truth]
        sample_truth = [truth[index] for index in indices]
        sample_predicted = [predicted[index] for index in indices]
        accuracy, macro_f1, _, _ = _point_metrics(sample_truth, sample_predicted)
        accuracies.append(accuracy)
        macro_scores.append(macro_f1)
    return (
        (float(np.quantile(accuracies, 0.025)), float(np.quantile(accuracies, 0.975))),
        (float(np.quantile(macro_scores, 0.025)), float(np.quantile(macro_scores, 0.975))),
    )


def _calibration(
    truth: list[str], predictions: Sequence[Prediction]
) -> tuple[float | None, float | None]:
    if not predictions or any(item.scores_by_class is None for item in predictions):
        return None, None
    score_rows = []
    for item in predictions:
        assert item.scores_by_class is not None
        score_rows.append([item.scores_by_class[SentimentLabel(label)] for label in LABELS])
    probabilities = np.array(score_rows)
    targets = np.array([[float(label == actual) for label in LABELS] for actual in truth])
    brier = float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))
    confidences = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets.argmax(axis=1)
    ece = 0.0
    bin_edges = np.linspace(0, 1, 11)
    for lower, upper in pairwise(bin_edges):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidences[mask].mean())
            )
    return ece, brier


def evaluate_predictions(
    truth: Sequence[SentimentLabel | str],
    predictions: Sequence[Prediction],
    *,
    languages: Sequence[str],
    bootstrap_samples: int = 500,
) -> EvaluationMetrics:
    if not truth or len(truth) != len(predictions) or len(truth) != len(languages):
        raise ValueError("truth, predictions, and languages must have equal non-zero length")
    truth_values = [SentimentLabel(value).value for value in truth]
    predicted_values = [
        item.predicted_label.value if item.predicted_label is not None else UNRESOLVED
        for item in predictions
    ]
    accuracy, macro_f1, matrix, per_class = _point_metrics(truth_values, predicted_values)
    coverage = sum(value != UNRESOLVED for value in predicted_values) / len(predicted_values)
    by_language: dict[str, dict[str, float | int]] = {}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, language in enumerate(languages):
        grouped[language].append(index)
    for language, indices in sorted(grouped.items()):
        lang_truth = [truth_values[index] for index in indices]
        lang_predicted = [predicted_values[index] for index in indices]
        lang_accuracy, lang_macro, _, _ = _point_metrics(lang_truth, lang_predicted)
        by_language[language] = {
            "support": len(indices),
            "accuracy": lang_accuracy,
            "macro_f1": lang_macro,
        }
    intervals = _bootstrap(truth_values, predicted_values, bootstrap_samples)
    ece, brier = _calibration(truth_values, predictions)
    latencies = np.array([item.latency_ms for item in predictions])
    provider_latencies: dict[str, dict[str, float | int]] = {}
    for provider in sorted({item.provider for item in predictions}):
        values = np.array([item.latency_ms for item in predictions if item.provider == provider])
        provider_latencies[provider] = {
            "support": int(values.size),
            "p50_ms": float(np.quantile(values, 0.5)),
            "p95_ms": float(np.quantile(values, 0.95)),
        }
    input_total = sum(item.usage_input_tokens or 0 for item in predictions)
    output_total = sum(item.usage_output_tokens or 0 for item in predictions)
    usage_count = sum(
        item.usage_input_tokens is not None or item.usage_output_tokens is not None
        for item in predictions
    )
    credits = [item.reported_credits for item in predictions if item.reported_credits is not None]
    priced_costs = [
        item.cost_usd_estimate for item in predictions if item.cost_usd_estimate is not None
    ]
    priced_fraction = len(priced_costs) / len(predictions)
    known_cost = float(sum(priced_costs))
    full_cost = known_cost if priced_fraction == 1 else None
    return EvaluationMetrics(
        support=len(predictions),
        accuracy=accuracy,
        macro_f1=macro_f1,
        operational_coverage=coverage,
        confusion_matrix=matrix,
        per_class=per_class,
        by_language=by_language,
        accuracy_ci95=intervals[0] if intervals else None,
        macro_f1_ci95=intervals[1] if intervals else None,
        ece=ece,
        brier_score=brier,
        latency_p50_ms=float(np.quantile(latencies, 0.5)),
        latency_p95_ms=float(np.quantile(latencies, 0.95)),
        latency_by_provider=provider_latencies,
        usage_input_tokens_total=input_total,
        usage_output_tokens_total=output_total,
        usage_records_fraction=usage_count / len(predictions),
        reported_credits_total=float(sum(credits)) if credits else None,
        cost_usd_known_total=known_cost,
        cost_usd_total=full_cost,
        cost_usd_per_1000=(full_cost * 1000 / len(predictions) if full_cost is not None else None),
        cost_priced_fraction=priced_fraction,
    )
