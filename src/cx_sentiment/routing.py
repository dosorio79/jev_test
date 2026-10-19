from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal

import polars as pl

from cx_sentiment.metrics import evaluate_predictions
from cx_sentiment.models import Prediction


class SelectionError(ValueError):
    """Threshold selection violated the validation-only protocol."""


@dataclass(frozen=True)
class HybridResult:
    prediction: Prediction
    route: Literal["decision", "fallback_low_score", "fallback_failure", "unresolved"]
    effective_latency_ms: float
    effective_cost_usd: float | None


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float | None
    macro_f1: float | None
    fallback_fraction: float | None
    expected_cost: float | None
    baseline_macro_f1: float
    max_f1_degradation: float
    frontier: list[dict[str, float | bool]]


def hybrid_prediction(
    decision: Prediction,
    fallback: Prediction | None,
    *,
    threshold: float,
    signal: Literal["max_probability", "provider_confidence"] = "max_probability",
) -> HybridResult:
    decision_cost = decision.cost_usd_estimate
    if signal == "max_probability":
        routing_score = (
            max(decision.scores_by_class.values()) if decision.scores_by_class is not None else None
        )
    else:
        routing_score = decision.provider_reported_confidence
    decision_succeeded = decision.predicted_label is not None
    if decision_succeeded and routing_score is not None and routing_score >= threshold:
        return HybridResult(decision, "decision", decision.latency_ms, decision_cost)
    if fallback is not None and fallback.predicted_label is not None:
        return HybridResult(
            fallback,
            "fallback_low_score" if decision_succeeded else "fallback_failure",
            decision.latency_ms + fallback.latency_ms,
            _combined_cost(decision_cost, fallback.cost_usd_estimate),
        )
    return HybridResult(decision, "unresolved", decision.latency_ms, decision_cost)


def _combined_cost(first: float | None, second: float | None) -> float | None:
    return first + second if first is not None and second is not None else None


def select_threshold(
    records: pl.DataFrame,
    decision_predictions: list[Prediction],
    fallback_predictions: list[Prediction],
    *,
    thresholds: Iterable[float] | None = None,
    max_f1_degradation: float,
    decision_cost: float | None = None,
    fallback_cost: float | None = None,
    signal: Literal["max_probability", "provider_confidence"] = "max_probability",
) -> ThresholdSelection:
    if records.is_empty() or set(records["source_split"].unique().to_list()) != {"validation"}:
        raise SelectionError("threshold selection requires only validation records")
    if len(decision_predictions) != records.height or len(fallback_predictions) != records.height:
        raise SelectionError("prediction counts must match validation records")
    truth = records["sentiment"].to_list()
    languages = records["language"].to_list()
    baseline = evaluate_predictions(
        truth, fallback_predictions, languages=languages, bootstrap_samples=0
    ).macro_f1
    grid = list(thresholds or (value / 100 for value in range(50, 100)))
    qualifying: list[ThresholdSelection] = []
    frontier: list[dict[str, float | bool]] = []
    for threshold in grid:
        hybrid = [
            hybrid_prediction(decision, fallback, threshold=threshold, signal=signal)
            for decision, fallback in zip(decision_predictions, fallback_predictions, strict=True)
        ]
        predictions = [result.prediction for result in hybrid]
        metric = evaluate_predictions(truth, predictions, languages=languages, bootstrap_samples=0)
        fallback_count = sum(result.route.startswith("fallback") for result in hybrid)
        fallback_fraction = fallback_count / records.height
        if decision_cost is not None and fallback_cost is not None:
            expected_cost = records.height * decision_cost + fallback_count * fallback_cost
        else:
            effective_costs = [result.effective_cost_usd for result in hybrid]
            if any(cost is None for cost in effective_costs):
                raise SelectionError(
                    "cached predictions need complete USD costs or explicit cost assumptions"
                )
            expected_cost = sum(cost for cost in effective_costs if cost is not None)
        qualifies = metric.macro_f1 >= baseline - max_f1_degradation
        frontier.append(
            {
                "threshold": threshold,
                "macro_f1": metric.macro_f1,
                "fallback_fraction": fallback_fraction,
                "coverage": 1 - fallback_fraction,
                "expected_cost_usd": expected_cost,
                "qualifies": qualifies,
            }
        )
        if qualifies:
            qualifying.append(
                ThresholdSelection(
                    threshold=threshold,
                    macro_f1=metric.macro_f1,
                    fallback_fraction=fallback_fraction,
                    expected_cost=expected_cost,
                    baseline_macro_f1=baseline,
                    max_f1_degradation=max_f1_degradation,
                    frontier=[],
                )
            )
    if not qualifying:
        return ThresholdSelection(
            threshold=None,
            macro_f1=None,
            fallback_fraction=None,
            expected_cost=None,
            baseline_macro_f1=baseline,
            max_f1_degradation=max_f1_degradation,
            frontier=frontier,
        )
    selected = min(qualifying, key=lambda item: (item.expected_cost, item.threshold))
    return replace(selected, frontier=frontier)
