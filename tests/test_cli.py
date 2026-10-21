from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import yaml

from cx_sentiment.budget import LiveSafety
from cx_sentiment.cli import main
from cx_sentiment.comparison import dataset_fingerprint
from cx_sentiment.config import load_config
from cx_sentiment.data import load_dataset
from cx_sentiment.providers.mock import MockProvider
from cx_sentiment.providers.openai_provider import OpenAIProvider


def test_reasoning_effort_is_loaded_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text("provider: openai\nreasoning_effort: none\n")
    assert load_config(config).reasoning_effort == "none"
    config.write_text("provider: openai\nreasoning_effort: unknown\n")
    with pytest.raises(ValueError, match="reasoning_effort"):
        load_config(config)


def test_reasoning_effort_separates_cached_runs(tmp_path: Path, monkeypatch) -> None:
    from cx_sentiment import cli

    calls = []
    reports = []

    def create(**kwargs):
        calls.append(kwargs.get("reasoning"))
        return SimpleNamespace(
            output_text='{"label":"neutral"}',
            model="gpt-6-luna",
            usage=None,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "require_quarto", lambda: None)
    monkeypatch.setattr(cli, "write_report", lambda *args, **kwargs: reports.append(kwargs))
    frame = pl.DataFrame(
        {
            "sample_id": ["s1"],
            "language": ["en"],
            "text": ["A statement."],
            "sentiment": ["neutral"],
        }
    )
    for effort in (None, "none", "none"):
        provider = OpenAIProvider(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            model="gpt-6-luna",
            reasoning_effort=effort,
        )
        cli._write_run(
            frame,
            provider,
            tmp_path / "report",
            LiveSafety(live=True, max_requests=1, max_estimated_cost_usd=1),
        )

    assert calls == [None, {"effort": "none"}]
    assert [report["metadata"]["reasoning_effort"] for report in reports] == [None, "none", "none"]


def test_offline_demo_writes_predictions_metrics_and_html_without_network(
    tmp_path: Path,
) -> None:
    output = tmp_path / "demo"
    exit_code = main(
        [
            "offline-demo",
            "--input",
            "examples/demo_feedback.csv",
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
                "examples/demo_feedback.csv",
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


def test_yaml_run_resolves_paths_relative_to_config(tmp_path: Path, monkeypatch) -> None:
    source = Path("examples/demo_feedback.csv").read_bytes()
    (tmp_path / "data.csv").write_bytes(source)
    config = tmp_path / "experiment.yaml"
    config.write_text("provider: mock\ninput: data.csv\noutput_dir: results\nlimit: 3\n")
    monkeypatch.chdir(tmp_path.parent)

    assert main(["run", str(config)]) == 0
    metrics = json.loads(next((tmp_path / "results").glob("*/metrics.json")).read_text())
    assert metrics["support"] == 3
    metadata = json.loads(next((tmp_path / "results").glob("*/report_metadata.json")).read_text())[
        "metadata"
    ]
    assert metadata["dataset_sha256"] == dataset_fingerprint(
        load_dataset(tmp_path / "data.csv").head(3)
    )


def test_yaml_live_provider_defaults_to_preview(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dataset = Path("examples/demo_feedback.csv").resolve()
    config = tmp_path / "experiment.yaml"
    config.write_text(f"provider: openai\ninput: {dataset}\noutput_dir: results\nlimit: 2\n")

    assert main(["run", str(config)]) == 0
    preview = next((tmp_path / "results").glob("*/dry_run.json"))
    plan = json.loads(preview.read_text())
    assert plan["network_calls"] == 0
    assert plan["would_process"] == 2
    assert str(preview) in capsys.readouterr().out


@pytest.mark.parametrize("content", ["limit: 0", "limti: 5", "[]", "live: true"])
def test_yaml_rejects_invalid_settings(tmp_path: Path, content: str, capsys) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(content)
    with pytest.raises(SystemExit) as error:
        main(["run", str(config)])
    assert error.value.code == 2
    assert "experiment.yaml" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("settings", "missing_field"),
    [("", "cost_per_request_usd"), ("cost_per_request_usd: 0.01\n", "budget_usd")],
)
def test_yaml_live_requires_budget_before_provider_creation(
    tmp_path: Path, monkeypatch, capsys, settings: str, missing_field: str
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dataset = Path("examples/demo_feedback.csv").resolve()
    config = tmp_path / "experiment.yaml"
    config.write_text(f"provider: openai\ninput: {dataset}\n{settings}")
    with pytest.raises(SystemExit) as error:
        main(["run", str(config), "--live"])
    assert error.value.code == 2
    assert missing_field in capsys.readouterr().err


@pytest.mark.parametrize("content", ["provider: [", "api_key: private-test-value"])
def test_yaml_errors_do_not_echo_config_values(tmp_path: Path, content: str, capsys) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(content)

    with pytest.raises(SystemExit) as error:
        main(["run", str(config)])

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "experiment.yaml" in message
    assert "private-test-value" not in message
    assert "Traceback" not in message


def test_yaml_missing_file_has_readable_error(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", str(tmp_path / "missing.yaml")])

    assert error.value.code == 2
    assert "missing.yaml" in capsys.readouterr().err


def test_yaml_run_prints_report_location(tmp_path: Path, capsys) -> None:
    dataset = Path("examples/demo_feedback.csv").resolve()
    config = tmp_path / "experiment.yaml"
    config.write_text(f"provider: mock\ninput: {dataset}\noutput_dir: results\nlimit: 1\n")

    assert main(["run", str(config)]) == 0
    report = next((tmp_path / "results").glob("*/report.html"))
    assert str(report) in capsys.readouterr().out


def test_yaml_cannot_be_mixed_with_legacy_flags(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text("provider: mock\n")
    with pytest.raises(SystemExit):
        main(["run", str(config), "--limit", "10"])


def test_yaml_live_run_passes_explicit_caps_to_runner(tmp_path: Path, monkeypatch) -> None:
    dataset = Path("examples/demo_feedback.csv").resolve()
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"provider: openai\ninput: {dataset}\nlimit: 2\nbudget_usd: 0.05\n"
        "cost_per_request_usd: 0.01\nreasoning_effort: none\n"
    )
    calls = []

    def capture_run(frame, provider, output_dir, safety, *, max_cost_per_request_usd):
        calls.append(safety)
        assert frame.height == 2
        assert safety.live is True
        assert safety.max_requests == 2
        assert safety.max_estimated_cost_usd == 0.05
        assert max_cost_per_request_usd == 0.01

    def capture_provider(name, *, reasoning_effort):
        assert name == "openai"
        assert reasoning_effort == "none"
        return MockProvider()

    monkeypatch.setattr("cx_sentiment.cli._provider_from_environment", capture_provider)
    monkeypatch.setattr("cx_sentiment.cli._write_run", capture_run)
    assert main(["run", str(config), "--live"]) == 0
    assert len(calls) == 1


def test_missing_quarto_stops_before_provider_calls(tmp_path: Path, monkeypatch, capsys) -> None:
    def unexpected_call(*args, **kwargs):
        pytest.fail("provider must not be called without Quarto")

    monkeypatch.setattr("cx_sentiment.reporting.shutil.which", lambda name: None)
    monkeypatch.setattr("cx_sentiment.cli.run_predictions", unexpected_call)
    config = tmp_path / "experiment.yaml"
    dataset = Path("examples/demo_feedback.csv").resolve()
    config.write_text(f"provider: mock\ninput: {dataset}\n")

    with pytest.raises(SystemExit) as error:
        main(["run", str(config)])

    assert error.value.code == 2
    assert "Quarto" in capsys.readouterr().err


def test_yaml_invocations_keep_separate_dated_runs_with_replayable_configs(
    tmp_path: Path, monkeypatch
) -> None:
    from cx_sentiment.config import load_config

    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    source = Path("examples/demo_feedback.csv").resolve()
    config = tmp_path / "smoke.yaml"
    config.write_text(f"provider: openai\ninput: {source}\noutput_dir: runs\nlimit: 5\n")

    assert main(["run", str(config)]) == 0
    assert main(["run", str(config)]) == 0

    runs = sorted((tmp_path / "runs").iterdir())
    assert len(runs) == 2
    for run in runs:
        assert run.name[:4].isdigit() and run.name[4] == "-"
        assert run.name.endswith("Z-openai-preview")
        snapshot = run / "experiment.yaml"
        assert "private-test-key" not in snapshot.read_text()
        archived = yaml.safe_load(snapshot.read_text())
        assert Path(archived["input"]) == source
        assert Path(archived["output_dir"]) == tmp_path / "runs"
        assert load_config(snapshot).limit == 5
        assert json.loads((run / "dry_run.json").read_text())["network_calls"] == 0

    # Replaying an archived YAML still writes a new sibling run.
    assert main(["run", str(runs[0] / "experiment.yaml")]) == 0
    assert len(list((tmp_path / "runs").iterdir())) == 3


def test_stopped_run_checkpoints_results_and_invocation_costs(tmp_path: Path, monkeypatch) -> None:
    from cx_sentiment import cli
    from cx_sentiment.budget import BudgetExceeded
    from cx_sentiment.models import Prediction, SentimentLabel

    output = tmp_path / "run"
    calls = []

    class Provider:
        provider_name = "fixture"
        model_id = "requested-model"
        prompt_version = "v1"
        requires_network = True

        def __init__(self):
            self.effective_settings = {"max_retries": 0, "price_card": {"as_of": "2026-10-18"}}

        def predict(self, text, language, sample_id):
            execution = json.loads((output / "execution.json").read_text())
            assert execution["complete"] is False
            assert execution["ledger"][len(calls)]["status"] == "running"
            assert (
                json.loads((output / "report_metadata.json").read_text())["metadata"]["complete"]
                is False
            )
            calls.append(sample_id)
            return Prediction(
                provider=self.provider_name,
                model_id="returned-model",
                returned_model_id="returned-model",
                prompt_version="v1",
                predicted_label=SentimentLabel.neutral,
                latency_ms=17,
                sample_id=sample_id,
                cost_usd_estimate=0.01,
            )

    frame = pl.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "language": ["en", "en"],
            "text": ["First", "Second"],
            "sentiment": ["neutral", "neutral"],
        }
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "require_quarto", lambda: None)
    with pytest.raises(BudgetExceeded):
        cli._write_run(
            frame,
            Provider(),
            output,
            LiveSafety(live=True, max_requests=1, max_estimated_cost_usd=1),
            max_cost_per_request_usd=0.02,
        )
    rows = (output / "predictions.jsonl").read_text().splitlines()
    assert len(rows) == 1
    execution = json.loads((output / "execution.json").read_text())
    assert execution["complete"] is False
    assert execution["stop_reason"] == "budget_exceeded"
    assert [row["status"] for row in execution["ledger"]] == ["success", "skipped"]
    assert execution["request_count"] == 1
    assert execution["known_spend_usd"] == 0.01
    assert execution["reserved_spend_usd"] == 0.02
    metadata = json.loads((output / "report_metadata.json").read_text())["metadata"]
    assert metadata["requested_model"] == "requested-model"
    assert metadata["returned_models"] == ["returned-model"]
    assert metadata["returned_model_unknown_records"] == 0
    assert metadata["pricing"]["as_of"] == "2026-10-18"
    assert metadata["package_version"]
    assert len(metadata["git_commit"]) == 40
    assert isinstance(metadata["git_dirty"], bool)
    assert metadata["complete"] is False
    assert not (output / "metrics.json").exists()


@pytest.mark.parametrize("failure", ["dataset", "credentials", "pricing"])
def test_expected_run_errors_are_sanitized(tmp_path: Path, monkeypatch, capsys, failure) -> None:
    dataset = Path("examples/demo_feedback.csv").resolve()
    if failure == "dataset":
        dataset = tmp_path / "invalid.csv"
        dataset.write_text("private-test-value\nsecret-input\n")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"provider: openai\ninput: {dataset}\nbudget_usd: 1\ncost_per_request_usd: 0.1\n"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if failure == "pricing":
        monkeypatch.setenv("OPENAI_API_KEY", "private-test-value")
        monkeypatch.setenv("OPENAI_MODEL", "private-test-value")
    with pytest.raises(SystemExit) as error:
        main(["run", str(config), "--live"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "private-test-value" not in message
    assert "Traceback" not in message
    assert {"dataset": "dataset", "credentials": "OPENAI_API_KEY", "pricing": "rate card"}[
        failure
    ] in message


def test_keyboard_interrupt_preserves_completed_results(tmp_path: Path, monkeypatch) -> None:
    from cx_sentiment import cli
    from cx_sentiment.models import Prediction, SentimentLabel

    class Provider:
        provider_name = "fixture"
        model_id = "fixture"
        prompt_version = "v1"
        requires_network = True

        def predict(self, text, language, sample_id):
            if sample_id == "s2":
                raise KeyboardInterrupt
            return Prediction(
                provider="fixture",
                model_id="fixture",
                prompt_version="v1",
                predicted_label=SentimentLabel.neutral,
                latency_ms=7,
                sample_id=sample_id,
                cost_usd_estimate=0.01,
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "require_quarto", lambda: None)
    frame = pl.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "language": ["en"] * 3,
            "text": ["First", "Second", "Third"],
            "sentiment": ["neutral"] * 3,
        }
    )
    with pytest.raises(KeyboardInterrupt):
        cli._write_run(
            frame,
            Provider(),
            tmp_path / "run",
            LiveSafety(live=True, max_requests=3, max_estimated_cost_usd=1),
            max_cost_per_request_usd=0.02,
        )
    execution = json.loads((tmp_path / "run/execution.json").read_text())
    assert execution["complete"] is False
    assert execution["stop_reason"] == "interrupted"
    assert execution["request_count"] == 2
    assert execution["unknown_cost_attempts"] == 1
    assert [row["status"] for row in execution["ledger"]] == ["success", "failure", "skipped"]
    assert len((tmp_path / "run/predictions.jsonl").read_text().splitlines()) == 1


def test_cache_hits_keep_historical_cost_out_of_invocation_spend(
    tmp_path: Path, monkeypatch
) -> None:
    from cx_sentiment import cli
    from cx_sentiment.models import Prediction, SentimentLabel

    class Provider:
        provider_name = "fixture"
        model_id = "fixture"
        prompt_version = "v1"
        requires_network = True

        def predict(self, text, language, sample_id):
            return Prediction(
                provider="fixture",
                model_id="fixture",
                prompt_version="v1",
                predicted_label=SentimentLabel.neutral,
                latency_ms=7,
                sample_id=sample_id,
                cost_usd_estimate=0.01,
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "require_quarto", lambda: None)
    monkeypatch.setattr(cli, "write_report", lambda *args, **kwargs: None)
    frame = pl.DataFrame(
        {
            "sample_id": ["s1"],
            "language": ["en"],
            "text": ["First"],
            "sentiment": ["neutral"],
        }
    )
    for name in ("first", "second"):
        cli._write_run(
            frame,
            Provider(),
            tmp_path / name,
            LiveSafety(live=True, max_requests=1, max_estimated_cost_usd=1),
            max_cost_per_request_usd=0.02,
        )
    execution = json.loads((tmp_path / "second/execution.json").read_text())
    assert execution["complete"] is True
    assert execution["request_count"] == 0
    assert execution["known_spend_usd"] == 0
    assert execution["reserved_spend_usd"] == 0
    assert execution["ledger"][0]["status"] == "cache_hit"
    saved = json.loads((tmp_path / "second/predictions.jsonl").read_text())
    assert saved["cost_usd_estimate"] == 0.01
    assert saved["latency_ms"] == 7
