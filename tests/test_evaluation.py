from __future__ import annotations

import polars as pl
import pytest

from cx_sentiment.metrics import evaluate_predictions
from cx_sentiment.models import Prediction, SentimentLabel
from cx_sentiment.routing import SelectionError, hybrid_prediction, select_threshold


def prediction(
    sample_id: str,
    label: SentimentLabel | None,
    scores: dict[SentimentLabel, float] | None = None,
    *,
    confidence: float | None = None,
    error: str | None = None,
    provider: str = "decision",
    latency_ms: float = 10,
    cost_usd: float | None = None,
    credits: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> Prediction:
    return Prediction(
        provider=provider,
        model_id="v1",
        prompt_version="p1",
        predicted_label=label,
        scores_by_class=scores,
        provider_reported_confidence=confidence,
        confidence_definition="provider" if confidence is not None else None,
        latency_ms=latency_ms,
        cost_usd_estimate=cost_usd,
        reported_credits=credits,
        usage_input_tokens=input_tokens,
        usage_output_tokens=output_tokens,
        error_kind=error,
        sample_id=sample_id,
    )


def test_hybrid_accepts_score_equal_to_threshold_and_falls_back_on_failure() -> None:
    scores = {
        SentimentLabel.negative: 0.1,
        SentimentLabel.neutral: 0.2,
        SentimentLabel.positive: 0.7,
    }
    decision = prediction("s1", SentimentLabel.positive, scores)
    fallback = prediction(
        "s1", SentimentLabel.neutral, provider="openai", latency_ms=100, cost_usd=0.004
    )
    accepted = hybrid_prediction(decision, fallback, threshold=0.7)
    failed = hybrid_prediction(prediction("s1", None, error="timeout"), fallback, threshold=0.7)

    assert accepted.prediction.predicted_label == SentimentLabel.positive
    assert accepted.route == "decision"
    assert accepted.effective_latency_ms == 10
    assert failed.prediction.predicted_label == SentimentLabel.neutral
    assert failed.route == "fallback_failure"
    assert failed.effective_latency_ms == 110
    assert failed.effective_cost_usd is None


def test_metrics_include_failures_in_denominator_and_calibration_only_with_scores() -> None:
    truth = [SentimentLabel.negative, SentimentLabel.positive]
    predictions = [
        prediction(
            "a",
            SentimentLabel.negative,
            {
                SentimentLabel.negative: 0.8,
                SentimentLabel.neutral: 0.1,
                SentimentLabel.positive: 0.1,
            },
        ),
        prediction("b", None, error="timeout"),
    ]
    result = evaluate_predictions(truth, predictions, languages=["en", "pt"], bootstrap_samples=20)

    assert result.accuracy == 0.5
    assert result.operational_coverage == 0.5
    assert result.ece is None
    assert result.brier_score is None
    assert result.by_language["pt"]["support"] == 1
    assert result.cost_usd_total is None
    assert result.cost_priced_fraction == 0


def test_metrics_report_latency_usage_cost_and_credits_by_provider() -> None:
    predictions = [
        prediction(
            "a",
            SentimentLabel.negative,
            provider="jev",
            latency_ms=20,
            cost_usd=0.001,
            credits=1,
            input_tokens=100,
            output_tokens=0,
        ),
        prediction(
            "b",
            SentimentLabel.positive,
            provider="openai",
            latency_ms=100,
            cost_usd=0.004,
            input_tokens=200,
            output_tokens=10,
        ),
    ]

    result = evaluate_predictions(
        [SentimentLabel.negative, SentimentLabel.positive],
        predictions,
        languages=["en", "pt"],
        bootstrap_samples=0,
    )

    assert result.latency_p50_ms == 60
    assert result.latency_p95_ms == pytest.approx(96)
    assert result.latency_by_provider["jev"]["p50_ms"] == 20
    assert result.latency_by_provider["openai"]["p95_ms"] == 100
    assert result.usage_input_tokens_total == 300
    assert result.usage_output_tokens_total == 10
    assert result.reported_credits_total == 1
    assert result.cost_usd_total == pytest.approx(0.005)
    assert result.cost_usd_per_1000 == pytest.approx(2.5)
    assert result.cost_priced_fraction == 1


def test_threshold_selection_rejects_test_split() -> None:
    records = pl.DataFrame(
        {
            "sample_id": ["s1"],
            "source_split": ["test"],
            "source_row_index_zero_based": [0],
            "language": ["en"],
            "text": ["text"],
            "sentiment": ["neutral"],
            "label": [1],
        }
    )
    with pytest.raises(SelectionError, match="validation"):
        select_threshold(records, [], [], max_f1_degradation=0.02)


def test_threshold_selection_chooses_lowest_cost_qualifying_policy() -> None:
    records = pl.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "source_split": ["validation"] * 3,
            "source_row_index_zero_based": [0, 1, 2],
            "language": ["en"] * 3,
            "text": ["text 0", "text 1", "text 2"],
            "sentiment": ["negative", "neutral", "positive"],
            "label": [0, 1, 2],
        }
    )
    decision = [
        prediction(
            f"s{i}",
            SentimentLabel(label),
            {
                SentimentLabel.negative: 0.8 if label == "negative" else 0.1,
                SentimentLabel.neutral: 0.8 if label == "neutral" else 0.1,
                SentimentLabel.positive: 0.8 if label == "positive" else 0.1,
            },
        )
        for i, label in enumerate(("negative", "neutral", "positive"))
    ]
    fallback = [
        prediction(f"s{i}", SentimentLabel(label), provider="openai")
        for i, label in enumerate(("negative", "neutral", "positive"))
    ]
    result = select_threshold(
        records,
        decision,
        fallback,
        thresholds=[0.5, 0.8, 0.9],
        max_f1_degradation=0.02,
        decision_cost=1,
        fallback_cost=10,
    )
    assert result.threshold == 0.5
    assert result.fallback_fraction == 0
    assert len(result.frontier) == 3
    assert result.frontier[0]["qualifies"] is True
