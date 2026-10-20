import json
from pathlib import Path

import polars as pl
import yaml

from cx_sentiment.cli import main
from cx_sentiment.models import Prediction, SentimentLabel


def _run(root: Path, provider: str) -> Path:
    directory = root / provider
    directory.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "source_split": ["validation"] * 3,
            "source_row_index_zero_based": [1, 2, 3],
            "language": ["en", "pt", "es"],
            "text": ["a", "b", "c"],
            "sentiment": ["negative", "neutral", "positive"],
            "label": [0, 1, 2],
        }
    )
    frame.write_csv(directory / "input.csv")
    (directory / "experiment.yaml").write_text(
        yaml.safe_dump({"input": "input.csv", "limit": 3}), encoding="utf-8"
    )
    predictions = []
    for sample_id, label in zip(frame["sample_id"], frame["sentiment"], strict=True):
        sentiment = SentimentLabel(label)
        scores = None
        if provider == "jev":
            scores = {item: 0.05 for item in SentimentLabel}
            scores[sentiment] = 0.9
        predictions.append(
            Prediction(
                provider=provider,
                model_id=f"{provider}-model",
                prompt_version=f"{provider}-prompt",
                predicted_label=sentiment,
                scores_by_class=scores,
                latency_ms=1,
                cost_usd_estimate=0.001,
                sample_id=sample_id,
            )
        )
    (directory / "predictions.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in predictions), encoding="utf-8"
    )
    (directory / "report_metadata.json").write_text(
        json.dumps({"metadata": {"complete": True}}), encoding="utf-8"
    )
    return directory


def test_offline_study_writes_result_without_provider_construction(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    jev = _run(tmp_path, "jev")
    openai = _run(tmp_path, "openai")
    config = tmp_path / "study.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "inputs": {"jev_run": str(jev), "openai_run": str(openai)},
                "output_dir": "outputs",
                "bootstrap_samples": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cx_sentiment.cli._provider_from_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("provider constructed")),
    )

    assert main(["study", str(config)]) == 0

    output = next((tmp_path / "outputs").iterdir())
    payload = json.loads((output / "study.json").read_text(encoding="utf-8"))
    assert len(payload["thresholds"]) == 60
    assert payload["measurement_scope"] == "simulated"
    assert "no provider calls" in capsys.readouterr().out


def test_live_study_without_live_flag_writes_zero_call_preview(
    tmp_path: Path, monkeypatch
) -> None:
    jev = _run(tmp_path, "jev")
    openai = _run(tmp_path, "openai")
    config = tmp_path / "study.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mode": "live",
                "inputs": {"jev_run": str(jev), "openai_run": str(openai)},
                "output_dir": "outputs",
                "live": {
                    "sample_size": 3,
                    "repetitions": 1,
                    "concurrency": 1,
                    "max_requests": 12,
                    "budget_usd": 1,
                    "cost_per_request_usd": 0.1,
                    "routing_threshold": 0.7,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cx_sentiment.cli._provider_from_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("provider constructed")),
    )

    assert main(["study", str(config)]) == 0

    output = next((tmp_path / "outputs").iterdir())
    preview = json.loads((output / "preview.json").read_text(encoding="utf-8"))
    assert preview["network_calls"] == 0
    assert preview["maximum_provider_requests"] == 12
