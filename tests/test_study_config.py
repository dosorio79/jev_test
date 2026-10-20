from pathlib import Path

import pytest

from cx_sentiment.study_config import load_study_config


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_offline_study_and_resolves_paths(tmp_path: Path) -> None:
    config = load_study_config(
        _write(
            tmp_path / "study.yaml",
            """
inputs:
  jev_run: saved/jev
  openai_run: saved/openai
  source_split: validation
output_dir: outputs
""",
        )
    )

    assert config.mode == "offline"
    assert config.inputs.jev_run == tmp_path / "saved/jev"
    assert config.inputs.openai_run == tmp_path / "saved/openai"
    assert config.output_dir == tmp_path / "outputs"
    assert config.thresholds == tuple(range(40, 100))
    assert config.live is None


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            "inputs: {jev_run: jev, openai_run: openai, source_split: test}\n",
            "source_split",
        ),
        (
            "inputs: {jev_run: jev, openai_run: openai, source_split: validation}\nunknown: 1\n",
            "unknown",
        ),
        (
            "inputs: {jev_run: jev, openai_run: openai, "
            "source_split: validation}\nquantiles: [0.5, 1.1]\n",
            "quantiles",
        ),
    ],
)
def test_rejects_invalid_offline_settings(tmp_path: Path, body: str, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        load_study_config(_write(tmp_path / "study.yaml", body))


@pytest.mark.parametrize("field", ["sample_size", "repetitions", "max_requests"])
def test_live_limits_must_be_positive(tmp_path: Path, field: str) -> None:
    live = {
        "sample_size": 10,
        "repetitions": 1,
        "concurrency": 1,
        "conditions": "[warm]",
        "max_requests": 30,
        "budget_usd": 1,
        "cost_per_request_usd": 0.1,
        "routing_threshold": 0.7,
    }
    live[field] = 0
    body = "\n".join(f"  {key}: {value}" for key, value in live.items())
    path = _write(
        tmp_path / "study.yaml",
        "mode: live\ninputs: {jev_run: jev, openai_run: openai, source_split: validation}\n"
        f"live:\n{body}\n",
    )

    with pytest.raises(ValueError, match=field):
        load_study_config(path)


def test_live_requires_budgets_and_supported_concurrency(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "study.yaml",
        """
mode: live
inputs: {jev_run: jev, openai_run: openai, source_split: validation}
live:
  sample_size: 10
  repetitions: 1
  concurrency: 2
  max_requests: 30
  routing_threshold: 0.7
""",
    )

    with pytest.raises(ValueError) as error:
        load_study_config(path)
    message = str(error.value)
    assert "concurrency" in message
    assert "budget_usd" in message
    assert "cost_per_request_usd" in message


def test_snapshot_excludes_credentials(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "study.yaml",
        """
inputs: {jev_run: jev, openai_run: openai, source_split: validation}
api_key: secret
""",
    )

    with pytest.raises(ValueError, match="api_key"):
        load_study_config(path)
