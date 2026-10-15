from __future__ import annotations

import json
from pathlib import Path

from cx_sentiment.cli import main


def test_offline_demo_writes_predictions_metrics_and_html_without_network(
    tmp_path: Path,
) -> None:
    output = tmp_path / "demo"
    exit_code = main(
        [
            "offline-demo",
            "--input",
            "data/cx_sentiment_test_600.csv",
            "--output-dir",
            str(output),
            "--limit",
            "12",
        ]
    )

    assert exit_code == 0
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 12
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["support"] == 12
    assert "Offline benchmark" in (output / "report.html").read_text()


def test_live_run_defaults_to_dry_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        main(
            [
                "run",
                "--provider",
                "openai",
                "--input",
                "data/cx_sentiment_test_600.csv",
                "--output-dir",
                str(tmp_path),
                "--limit",
                "5",
            ]
        )
        == 0
    )
    plan = json.loads((tmp_path / "dry_run.json").read_text())
    assert plan["network_calls"] == 0
    assert plan["would_process"] == 5
