import csv
import json
from pathlib import Path

from cx_sentiment.metrics import evaluate_predictions
from cx_sentiment.models import Prediction, SentimentLabel
from cx_sentiment.reporting import write_report


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
        metadata={"note": "<script>alert(1)</script>"},
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
    assert "<script>" not in document
    assert "partial ($0.000000 known; 0.0% priced)" in document
