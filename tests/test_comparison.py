from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import yaml

from cx_sentiment.comparison import (
    CostEstimate,
    compare_runs,
    dataset_fingerprint,
    load_comparison_config,
)
from cx_sentiment.data import load_dataset
from cx_sentiment.models import Prediction, SentimentLabel


@pytest.fixture
def saved_runs(tmp_path: Path) -> Path:
    source = Path("examples/demo_feedback.csv").resolve()
    frame = load_dataset(source).head(3)
    for name in ("first", "second"):
        run = tmp_path / name
        run.mkdir()
        (run / "experiment.yaml").write_text(
            yaml.safe_dump(
                {
                    "provider": "mock",
                    "input": str(source),
                    "limit": 3,
                }
            )
        )
        predictions = [
            Prediction(
                provider="mock",
                model_id="fixture",
                prompt_version="test",
                sample_id=sample_id,
                predicted_label=SentimentLabel(truth),
                latency_ms=10,
                cost_usd_estimate=0,
            )
            for sample_id, truth in frame.select("sample_id", "sentiment").iter_rows()
        ]
        (run / "predictions.jsonl").write_text(
            "".join(p.model_dump_json() + "\n" for p in predictions)
        )
        (run / "report_metadata.json").write_text(
            json.dumps(
                {
                    "metadata": {"dataset_sha256": dataset_fingerprint(frame)},
                }
            )
        )
    path = tmp_path / "compare.yaml"
    path.write_text("title: Example\nruns: [first, second]\n")
    return path


def test_comparison_aligns_predictions_and_recalculates_metrics(saved_runs: Path) -> None:
    predictions = saved_runs.parent / "second/predictions.jsonl"
    rows = predictions.read_text().splitlines()
    changed = json.loads(rows[0])
    changed["predicted_label"] = (
        "neutral" if changed["predicted_label"] != "neutral" else "positive"
    )
    rows[0] = json.dumps(changed)
    predictions.write_text("\n".join(reversed(rows)) + "\n")
    result = compare_runs(load_comparison_config(saved_runs))
    assert result["runs"][0]["metrics"]["accuracy"] == 1
    assert result["runs"][1]["metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert len(result["disagreements"]) == 1
    disagreement = result["disagreements"][0]
    reference = load_dataset(Path("examples/demo_feedback.csv"))
    expected = reference.filter(pl.col("sample_id") == disagreement["sample_id"])["text"][0]
    assert disagreement["verbatim"] == expected
    assert result["warnings"] == []
    assert result["historical_inputs_verified"] is True


@pytest.mark.parametrize("provider", ["ollama", "hybrid"])
def test_comparison_reads_saved_experiment_snapshots(saved_runs: Path, provider: str) -> None:
    run = saved_runs.parent / "second"
    snapshot = yaml.safe_load((run / "experiment.yaml").read_text())
    snapshot.update(provider=provider, model="local-model", think=False, threshold=0.72)
    (run / "experiment.yaml").write_text(yaml.safe_dump(snapshot))
    electricity = {
        "tariff_per_kwh": 0.18,
        "rows": [
            {"Scope": "GPU only", "Basis": "Sampled", "Energy kWh": 0.01, "Estimated EUR": 0.0018}
        ],
        "scope_note": "GPU only; excludes the rest of the computer.",
    }
    report = json.loads((run / "report_metadata.json").read_text())
    report["metadata"].update(electricity_estimate=electricity, think=False, threshold=0.72)
    (run / "report_metadata.json").write_text(json.dumps(report))
    predictions = [
        Prediction.model_validate_json(line).model_copy(
            update={"provider": provider, "cost_usd_estimate": None}
        )
        for line in (run / "predictions.jsonl").read_text().splitlines()
    ]
    (run / "predictions.jsonl").write_text("".join(p.model_dump_json() + "\n" for p in predictions))

    result = compare_runs(load_comparison_config(saved_runs))

    assert result["historical_inputs_verified"] is True
    assert result["runs"][1]["provider"] == provider
    assert result["runs"][1]["metrics"]["cost_usd_total"] is None
    assert result["runs"][1]["electricity_estimate"] == electricity
    assert result["runs"][1]["think"] is False
    assert result["runs"][1]["threshold"] == 0.72


def test_comparison_cost_estimate_preserves_unpriced_predictions(saved_runs: Path) -> None:
    path = saved_runs.parent / "second/predictions.jsonl"
    predictions = [
        Prediction.model_validate_json(line).model_copy(update={"cost_usd_estimate": None})
        for line in path.read_text().splitlines()
    ]
    path.write_text("".join(p.model_dump_json() + "\n" for p in predictions))
    original = path.read_bytes()
    config = load_comparison_config(saved_runs).model_copy(
        update={
            "cost_estimates": {"Run 2": CostEstimate(usd_total=0.007, basis="Electricity ballpark")}
        }
    )

    result = compare_runs(config)

    run = result["runs"][1]
    assert run["metrics"]["cost_usd_total"] is None
    assert run["metrics"]["cost_priced_fraction"] == 0
    assert run["comparison_cost_usd"] == 0.007
    assert run["cost_estimate"]["basis"] == "Electricity ballpark"
    assert path.read_bytes() == original


def test_comparison_rejects_estimate_for_priced_run(saved_runs: Path) -> None:
    config = load_comparison_config(saved_runs).model_copy(
        update={
            "cost_estimates": {"Run 2": CostEstimate(usd_total=0.007, basis="Electricity ballpark")}
        }
    )
    with pytest.raises(ValueError, match="completely unpriced"):
        compare_runs(config)


@pytest.mark.parametrize("change", ["duplicate", "missing", "foreign"])
def test_comparison_rejects_invalid_prediction_ids(saved_runs: Path, change: str) -> None:
    path = saved_runs.parent / "second/predictions.jsonl"
    rows = path.read_text().splitlines()
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "missing":
        rows.pop()
    else:
        item = json.loads(rows[0])
        item["sample_id"] = "foreign"
        rows[0] = json.dumps(item)
    path.write_text("\n".join(rows))
    with pytest.raises(ValueError, match="sample IDs"):
        compare_runs(load_comparison_config(saved_runs))


def test_comparison_rejects_changed_source_data(saved_runs: Path) -> None:
    run = saved_runs.parent / "second"
    frame = load_dataset(Path("examples/demo_feedback.csv")).head(3)
    altered = frame.with_columns(pl.lit("Changed input").alias("text"))
    source = saved_runs.parent / "changed.csv"
    altered.write_csv(source)
    (run / "experiment.yaml").write_text(
        yaml.safe_dump(
            {
                "provider": "mock",
                "input": str(source),
                "limit": 3,
            }
        )
    )
    with pytest.raises(ValueError, match=r"fingerprint|source records"):
        compare_runs(load_comparison_config(saved_runs))


def test_comparison_rejects_fingerprint_mismatch(saved_runs: Path) -> None:
    path = saved_runs.parent / "first/report_metadata.json"
    path.write_text('{"metadata": {"dataset_sha256": "different"}}')
    with pytest.raises(ValueError, match="fingerprint"):
        compare_runs(load_comparison_config(saved_runs))


def test_legacy_runs_have_explicit_historical_warning(saved_runs: Path) -> None:
    for name in ("first", "second"):
        (saved_runs.parent / name / "report_metadata.json").write_text('{"metadata": {}}')
    result = compare_runs(load_comparison_config(saved_runs))
    assert not result["historical_inputs_verified"]
    assert len(result["warnings"]) == 2
    assert "historical" in result["warnings"][0]


@pytest.mark.parametrize(
    "settings",
    ["runs: [first]", "runs: [first, first]", "runs: [first, second]\nsecret: value", "[]"],
)
def test_invalid_comparison_settings(tmp_path: Path, settings: str) -> None:
    path = tmp_path / "compare.yaml"
    path.write_text(settings)
    with pytest.raises(ValueError):
        load_comparison_config(path)


def test_preview_is_not_a_completed_run(saved_runs: Path) -> None:
    (saved_runs.parent / "first/predictions.jsonl").unlink()
    with pytest.raises(ValueError, match="cannot read"):
        compare_runs(load_comparison_config(saved_runs))


@pytest.mark.parametrize("pricing", ["partial", "unavailable", "electricity"])
def test_compare_cli_renders_without_providers(
    saved_runs: Path, monkeypatch, capsys, pricing: str
) -> None:
    from cx_sentiment.cli import main

    def unexpected_provider(*args, **kwargs):
        raise AssertionError("comparison must not construct providers")

    monkeypatch.setattr("cx_sentiment.cli._provider_from_environment", unexpected_provider)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    (saved_runs.parent / "first/execution.json").write_text(
        json.dumps(
            {
                "complete": True,
                "request_count": 2,
                "known_spend_usd": 0.001,
                "unknown_cost_attempts": 1,
                "reserved_spend_usd": 0.02,
            }
        )
    )
    # Keep source feedback exact, including Unicode, newlines, and HTML-like text.
    verbatim = "Feedback <script>alert('verbatim')</script>\nSecond line — ação"
    frame = load_dataset(Path("examples/demo_feedback.csv")).head(3)
    frame = frame.with_columns(
        pl.when(pl.col("sample_id") == frame["sample_id"][0])
        .then(pl.lit(verbatim))
        .otherwise(pl.col("text"))
        .alias("text")
    )
    source = saved_runs.parent / "feedback.csv"
    frame.write_csv(source)
    for name in ("first", "second"):
        experiment_path = saved_runs.parent / name / "experiment.yaml"
        experiment = yaml.safe_load(experiment_path.read_text())
        experiment["input"] = str(source)
        experiment_path.write_text(yaml.safe_dump(experiment))
        metadata_path = saved_runs.parent / name / "report_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["metadata"]["dataset_sha256"] = dataset_fingerprint(frame)
        metadata_path.write_text(json.dumps(metadata))
    # Exercise incomplete costs and an unresolved prediction in an actual render.
    originals = {}
    for name in ("first", "second"):
        path = saved_runs.parent / name / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["cost_usd_estimate"] = 0.01 if pricing == "partial" else None
            if name == "second":
                row["provider"] = "hybrid"
                row["model_id"] = "jev-1.13.0 -> gpt-6-luna"
        if name == "second":
            rows[0]["cost_usd_estimate"] = None
            rows[0]["predicted_label"] = None
            rows[0]["error_kind"] = "timeout"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        originals[path] = path.read_bytes()
    metadata_path = saved_runs.parent / "second/report_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["metadata"]["cost_and_usage_scope"] = "Includes rejected attempts and recollection."
    metadata["metadata"]["latency_scope"] = "Original rejected-attempt latency unavailable."
    metadata_path.write_text(json.dumps(metadata))
    (saved_runs.parent / "second/recollection_audit.json").write_text(
        json.dumps([{"fresh_cost_usd": 0.001}])
    )
    saved_runs.write_text('title: "<img src=x onerror=alert(1)>"\nruns: [first, second]\n')
    if pricing == "electricity":
        with saved_runs.open("a") as handle:
            handle.write(
                "cost_estimates:\n  Run 2:\n    usd_total: 0.007\n    basis: Electricity ballpark\n"
            )
    assert main(["compare", str(saved_runs)]) == 0
    output = next((saved_runs.parent / "runs").glob("*-comparison"))
    result = json.loads((output / "comparison.json").read_text())
    assert result["runs"][0]["metrics"]["cost_usd_total"] == (
        pytest.approx(0.03) if pricing == "partial" else None
    )
    assert result["runs"][1]["metrics"]["cost_usd_total"] is None
    assert result["runs"][1]["metrics"]["operational_coverage"] == pytest.approx(2 / 3)
    assert result["disagreements"][0]["Run 2"] == "unresolved"
    assert result["disagreements"][0]["verbatim"] == verbatim
    html = (output / "report.html").read_text()
    assert "Quality versus cost and latency" in html
    assert "Macro F1 versus P50 latency (typical request)" in html
    assert "Macro F1 versus P95 latency (slower tail)" in html
    assert "Jev + Luna fallback" in html
    assert '"name":"fixture"' in html
    assert '"name":"Jev + Luna fallback"' in html
    assert "Run 1: mock" not in html
    assert "Run 2: hybrid" not in html
    assert "not a paired speedup measurement" in html
    assert "Cost accounting" in html
    assert "Source invocation accounting" in html
    assert "Known invocation USD" in html
    assert result["runs"][0]["invocation_accounting"]["request_count"] == 2
    assert "Additional request USD" in html
    assert "&lt;script&gt;" in html
    assert "Second line" in html
    assert "<script>alert('verbatim')</script>" not in html
    assert "Includes rejected attempts and recollection." in html
    assert "Original rejected-attempt latency unavailable." in html
    assert "do not add their totals together" in html
    if pricing == "partial":
        assert "Partial (lower bound)" in html
        assert "Comparison USD per 1,000" in html
    elif pricing == "electricity":
        assert "Electricity estimate" in html
        assert "0.007000" in html
        assert result["runs"][1]["comparison_cost_usd"] == 0.007
        assert result["runs"][1]["metrics"]["cost_usd_total"] is None
    else:
        assert "No pricing available" in html
    assert "plotly" in html.lower()
    assert "&lt;img" in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert (output / "report.qmd").is_file()
    replay = load_comparison_config(output / "comparison.yaml")
    assert replay.output_dir == saved_runs.parent / "runs"
    assert replay.runs == load_comparison_config(saved_runs).runs
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert "no provider calls" in capsys.readouterr().out


def test_compare_cli_reports_readable_errors(tmp_path: Path, capsys) -> None:
    from cx_sentiment.cli import main

    path = tmp_path / "compare.yaml"
    path.write_text("runs: [missing-one, missing-two]\n")
    with pytest.raises(SystemExit) as error:
        main(["compare", str(path)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "cannot read" in message
    assert "Traceback" not in message


def test_comparison_preserves_collection_cost_and_latency_context(saved_runs: Path) -> None:
    path = saved_runs.parent / "second/report_metadata.json"
    report = json.loads(path.read_text())
    report["title"] = "Jev after bounded recollection"
    report["metadata"].update(
        {
            "cost_and_usage_scope": "Includes original rejected attempts and recollection.",
            "latency_scope": "Original rejected-attempt latency unavailable.",
            "collection_policy": "One fresh attempt per improper output.",
        }
    )
    path.write_text(json.dumps(report))
    result = compare_runs(load_comparison_config(saved_runs))
    run = result["runs"][1]
    assert run["title"] == report["title"]
    assert run["cost_and_usage_scope"] == report["metadata"]["cost_and_usage_scope"]
    assert run["latency_scope"] == report["metadata"]["latency_scope"]
    assert run["collection_policy"] == report["metadata"]["collection_policy"]


def test_comparison_reports_explicit_recollection_spend(saved_runs: Path) -> None:
    (saved_runs.parent / "second/recollection_audit.json").write_text(
        json.dumps(
            [
                {"fresh_cost_usd": 0.001},
                {"fresh_cost_usd": 0.002},
            ]
        )
    )
    result = compare_runs(load_comparison_config(saved_runs))
    assert result["runs"][0]["recollection_cost_usd"] is None
    assert result["runs"][1]["recollection_cost_usd"] == pytest.approx(0.003)
    assert result["runs"][1]["recollection_attempts"] == 2


@pytest.mark.parametrize("cost", [-1, float("nan"), "unknown", True])
def test_comparison_rejects_invalid_recollection_cost(saved_runs: Path, cost) -> None:
    (saved_runs.parent / "second/recollection_audit.json").write_text(
        json.dumps([{"fresh_cost_usd": cost}])
    )
    with pytest.raises(ValueError, match="invalid saved"):
        compare_runs(load_comparison_config(saved_runs))


@pytest.mark.parametrize("source", ["metadata", "execution"])
def test_comparison_rejects_incomplete_runs_before_scoring(saved_runs: Path, source: str) -> None:
    directory = saved_runs.parent / "first"
    if source == "metadata":
        path = directory / "report_metadata.json"
        report = json.loads(path.read_text())
        report["metadata"]["complete"] = False
        path.write_text(json.dumps(report))
    else:
        (directory / "execution.json").write_text(json.dumps({"complete": False}))
    (directory / "predictions.jsonl").write_text("")
    with pytest.raises(ValueError, match="incomplete"):
        compare_runs(load_comparison_config(saved_runs))
