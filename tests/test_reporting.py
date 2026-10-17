import csv
import json
import subprocess
from pathlib import Path

import pytest

from cx_sentiment.metrics import evaluate_predictions
from cx_sentiment.models import Prediction, SentimentLabel
from cx_sentiment.reporting import write_report
from cx_sentiment.routing import ThresholdSelection


def test_report_preserves_formats_missing_costs_and_escapes_html(tmp_path: Path) -> None:
    prediction = Prediction(
        provider="<provider>",
        model_id="test",
        prompt_version="v1",
        predicted_label=SentimentLabel.positive,
        latency_ms=12,
        sample_id="s1",
    )
    metrics = evaluate_predictions(
        ["positive"],
        [prediction],
        languages=["en"],
        bootstrap_samples=0,
    )

    write_report(
        tmp_path,
        [prediction],
        metrics,
        title="<benchmark>",
        metadata={
            "note": "<script>alert(1)</script>",
            "electricity_estimate": {
                "tariff_per_kwh": 0.18,
                "rows": [
                    {
                        "Scope": "<GPU>",
                        "Basis": "Sampled board power",
                        "Energy kWh": 0.01,
                        "Estimated EUR": 0.0018,
                    }
                ],
                "scope_note": "GPU only; excludes other laptop components.",
            },
        },
    )

    assert (
        Prediction.model_validate_json((tmp_path / "predictions.jsonl").read_text()) == prediction
    )
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics.model_dump(mode="json")
    with (tmp_path / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["scope"] == "overall"
    assert rows[0]["coverage"] == "1.0"
    assert rows[0]["cost_usd_total"] == ""
    assert rows[1]["scope"] == "en"
    assert rows[1]["coverage"] == ""
    document = (tmp_path / "report.html").read_text()
    assert "&lt;benchmark&gt;" in document
    assert "&lt;provider&gt;" in document
    assert "&lt;script&gt;" in document
    assert "<script>alert(1)</script>" not in document
    assert "USD cost unavailable: no verified prices were supplied." in document
    assert "Electricity-only estimates at EUR 0.18/kWh" in document
    assert "&lt;GPU&gt;" in document
    assert "Estimated EUR" in document
    assert "0.001800" in document
    assert "GPU only; excludes other laptop components." in document
    assert (tmp_path / "report.qmd").is_file()
    assert json.loads((tmp_path / "report_metadata.json").read_text())["title"] == "<benchmark>"
    assert "Plotly.newPlot" in document


@pytest.mark.parametrize("state", ["empty", "selected", "no_qualifying"])
def test_selection_report_renders_saved_data(tmp_path: Path, state: str) -> None:
    from cx_sentiment.reporting import write_selection_report

    path = tmp_path / "selection with spaces.json"
    selected = state == "selected"
    selection = ThresholdSelection(
        threshold=0.7 if selected else None,
        macro_f1=0.9 if selected else None,
        fallback_fraction=0.2 if selected else None,
        expected_cost=0.01 if selected else None,
        baseline_macro_f1=0.9,
        max_f1_degradation=0.02,
        frontier=[
            {
                "threshold": 0.7,
                "macro_f1": 0.9 if selected else 0.5,
                "fallback_fraction": 0.2,
                "coverage": 0.8,
                "expected_cost_usd": 0.01,
                "qualifies": selected,
            }
        ]
        if state != "empty"
        else [],
    )

    write_selection_report(path, selection)

    assert path.with_suffix(".qmd").is_file()
    assert json.loads(path.read_text())["threshold"] == selection.threshold
    document = path.with_suffix(".html").read_text()
    assert "Routing threshold frontier" in document
    if state != "empty":
        assert "Plotly.newPlot" in document
        assert "Expected cost (USD)" in document
    if not selected:
        assert "No threshold" in document


@pytest.mark.parametrize("failure", ["missing", "timeout", "exit"])
def test_quarto_failure_preserves_saved_results(tmp_path: Path, monkeypatch, failure: str) -> None:
    monkeypatch.setattr(
        "cx_sentiment.reporting.shutil.which",
        lambda name: None if failure == "missing" else "quarto",
    )

    def fail_render(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("quarto", 120)
        raise subprocess.CalledProcessError(1, "quarto")

    monkeypatch.setattr("cx_sentiment.reporting.subprocess.run", fail_render)
    prediction = Prediction(
        provider="mock",
        model_id="v1",
        prompt_version="p1",
        predicted_label=SentimentLabel.positive,
        latency_ms=0,
        sample_id="s1",
    )
    metrics = evaluate_predictions(
        ["positive"],
        [prediction],
        languages=["en"],
        bootstrap_samples=0,
    )

    with pytest.raises(RuntimeError, match="Quarto"):
        write_report(tmp_path, [prediction], metrics, title="Example")

    assert (tmp_path / "predictions.jsonl").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "report.qmd").is_file()
