import json
from pathlib import Path

import polars as pl
import pytest
import yaml

from cx_sentiment.models import Prediction, SentimentLabel
from cx_sentiment.study import analyze_saved_validation
from cx_sentiment.study_config import OfflineInputs, StudyConfig


def _prediction(sample_id: str, label: str, confidence: float, provider: str) -> Prediction:
    predicted = SentimentLabel(label)
    scores = None
    if provider == "jev":
        remainder = (1 - confidence) / 2
        scores = {item: remainder for item in SentimentLabel}
        scores[predicted] = confidence
    return Prediction(
        provider=provider,
        model_id=f"{provider}-model",
        prompt_version=f"{provider}-prompt",
        predicted_label=predicted,
        scores_by_class=scores,
        latency_ms=10,
        cost_usd_estimate=0.001 if provider == "jev" else 0.01,
        sample_id=sample_id,
    )


def _saved_run(
    root: Path,
    provider: str,
    rows: list[tuple[str, str, str, float]],
    *,
    split: str = "validation",
    complete: bool = True,
) -> Path:
    directory = root / provider
    directory.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "sample_id": [row[0] for row in rows],
            "source_split": [split] * len(rows),
            "source_row_index_zero_based": list(range(len(rows))),
            "language": [row[1] for row in rows],
            "text": [f"text {index}" for index in range(len(rows))],
            "sentiment": [row[2] for row in rows],
            "label": [{"negative": 0, "neutral": 1, "positive": 2}[row[2]] for row in rows],
        }
    )
    frame.write_csv(directory / "input.csv")
    (directory / "experiment.yaml").write_text(
        yaml.safe_dump({"input": "input.csv", "limit": len(rows)}), encoding="utf-8"
    )
    predictions = [
        _prediction(sample_id, label, confidence, provider)
        for sample_id, _language, label, confidence in rows
    ]
    (directory / "predictions.jsonl").write_text(
        "\n".join(item.model_dump_json() for item in predictions) + "\n", encoding="utf-8"
    )
    (directory / "report_metadata.json").write_text(
        json.dumps({"metadata": {"complete": complete}}), encoding="utf-8"
    )
    return directory


def test_threshold_sweep_reports_quality_coverage_cost_and_languages(tmp_path: Path) -> None:
    rows = [
        ("s1", "en", "negative", 0.4),
        ("s2", "en", "neutral", 0.7),
        ("s3", "pt", "positive", 0.99),
    ]
    jev = _saved_run(tmp_path, "jev", rows)
    openai = _saved_run(tmp_path, "openai", rows)
    config = StudyConfig(
        inputs=OfflineInputs(jev_run=jev, openai_run=openai), bootstrap_samples=20
    )

    result = analyze_saved_validation(config)

    assert len(result.thresholds) == 60
    point = result.thresholds[0]
    assert point.threshold == 0.4
    assert point.jev_coverage == 1
    assert point.fallback_fraction == 0
    assert point.accuracy == 1
    assert point.macro_f1 == 1
    assert point.macro_f1_delta_from_openai == 0
    assert point.accepted_error_rate == 0
    assert point.deferred_error_rate is None
    assert point.cost_usd_per_1000 == pytest.approx(1)
    assert {item.language for item in point.by_language} == {"en", "pt"}


def test_analysis_rejects_non_validation_and_incomplete_runs(tmp_path: Path) -> None:
    rows = [("s1", "en", "negative", 0.8)]
    jev = _saved_run(tmp_path, "jev", rows, split="test")
    openai = _saved_run(tmp_path, "openai", rows, split="test")
    config = StudyConfig(inputs=OfflineInputs(jev_run=jev, openai_run=openai))
    with pytest.raises(ValueError, match="validation"):
        analyze_saved_validation(config)

    complete_root = tmp_path / "incomplete"
    complete_root.mkdir()
    jev = _saved_run(complete_root, "jev", rows, complete=False)
    openai = _saved_run(complete_root, "openai", rows)
    config = StudyConfig(inputs=OfflineInputs(jev_run=jev, openai_run=openai))
    with pytest.raises(ValueError, match="incomplete"):
        analyze_saved_validation(config)


def test_analysis_rejects_mismatched_ids_and_missing_probabilities(tmp_path: Path) -> None:
    rows = [("s1", "en", "negative", 0.8)]
    jev = _saved_run(tmp_path, "jev", rows)
    openai = _saved_run(tmp_path, "openai", [("different", "en", "negative", 0.8)])
    config = StudyConfig(inputs=OfflineInputs(jev_run=jev, openai_run=openai))
    with pytest.raises(ValueError, match="identical source records"):
        analyze_saved_validation(config)

    prediction = _prediction("s1", "negative", 0.8, "openai").model_copy(
        update={"provider": "jev"}
    )
    (jev / "predictions.jsonl").write_text(prediction.model_dump_json() + "\n", encoding="utf-8")
    openai = _saved_run(tmp_path / "second", "openai", rows)
    config = StudyConfig(inputs=OfflineInputs(jev_run=jev, openai_run=openai))
    with pytest.raises(ValueError, match="probability"):
        analyze_saved_validation(config)
