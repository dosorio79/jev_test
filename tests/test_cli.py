from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from cx_sentiment.cli import main
from cx_sentiment.providers.mock import MockProvider


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


def test_yaml_run_resolves_paths_relative_to_config(tmp_path: Path, monkeypatch) -> None:
    source = Path("data/cx_sentiment_test_600.csv").read_bytes()
    (tmp_path / "data.csv").write_bytes(source)
    config = tmp_path / "experiment.yaml"
    config.write_text("provider: mock\ninput: data.csv\noutput_dir: results\nlimit: 3\n")
    monkeypatch.chdir(tmp_path.parent)

    assert main(["run", str(config)]) == 0
    metrics = json.loads(next((tmp_path / "results").glob("*/metrics.json")).read_text())
    assert metrics["support"] == 3


def test_yaml_live_provider_defaults_to_preview(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dataset = Path("data/cx_sentiment_test_600.csv").resolve()
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
    dataset = Path("data/cx_sentiment_test_600.csv").resolve()
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
    dataset = Path("data/cx_sentiment_test_600.csv").resolve()
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
    dataset = Path("data/cx_sentiment_test_600.csv").resolve()
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"provider: jev\ninput: {dataset}\nlimit: 2\nbudget_usd: 0.05\ncost_per_request_usd: 0.01\n"
    )
    calls = []

    def capture_run(frame, provider, output_dir, safety, *, max_cost_per_request_usd):
        calls.append(safety)
        assert frame.height == 2
        assert safety.live is True
        assert safety.max_requests == 2
        assert safety.max_estimated_cost_usd == 0.05
        assert max_cost_per_request_usd == 0.01

    monkeypatch.setattr("cx_sentiment.cli._provider_from_environment", lambda name: MockProvider())
    monkeypatch.setattr("cx_sentiment.cli._write_run", capture_run)
    assert main(["run", str(config), "--live"]) == 0
    assert len(calls) == 1


def test_missing_quarto_stops_before_provider_calls(tmp_path: Path, monkeypatch, capsys) -> None:
    def unexpected_call(*args, **kwargs):
        pytest.fail("provider must not be called without Quarto")

    monkeypatch.setattr("cx_sentiment.reporting.shutil.which", lambda name: None)
    monkeypatch.setattr("cx_sentiment.cli.run_predictions", unexpected_call)
    config = tmp_path / "experiment.yaml"
    dataset = Path("data/cx_sentiment_test_600.csv").resolve()
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
    source = Path("data/cx_sentiment_test_600.csv").resolve()
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
