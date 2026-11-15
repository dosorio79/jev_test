import time
from pathlib import Path

import polars as pl
import pytest

from cx_sentiment.benchmark import build_schedule, run_live_benchmark
from cx_sentiment.budget import LiveSafety
from cx_sentiment.models import Prediction, SentimentLabel


class FakeProvider:
    requires_network = True

    def __init__(self, name: str, label: SentimentLabel, score: float | None = None) -> None:
        self.provider_name = name
        self.model_id = f"{name}-model"
        self.prompt_version = f"{name}-prompt"
        self.label = label
        self.score = score
        self.calls = 0

    def predict(self, text: str, language: str, sample_id: str) -> Prediction:
        self.calls += 1
        scores = None
        if self.score is not None:
            remainder = (1 - self.score) / 2
            scores = {label: remainder for label in SentimentLabel}
            scores[self.label] = self.score
        return Prediction(
            provider=self.provider_name,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            predicted_label=self.label,
            scores_by_class=scores,
            latency_ms=5,
            cost_usd_estimate=0.01,
            sample_id=sample_id,
        )


def _records() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "language": ["en", "en", "pt", "pt"],
            "sentiment": ["negative", "positive", "negative", "positive"],
            "text": ["a", "b", "c", "d"],
        }
    )


def test_schedule_is_paired_reproducible_and_rotates_strategy_order() -> None:
    first = build_schedule(_records(), sample_size=4, repetitions=2, seed=7)
    second = build_schedule(_records(), sample_size=4, repetitions=2, seed=7)

    assert first == second
    assert {item.sample_id for item in first} == {"a", "b", "c", "d"}
    assert len(first) == 4 * 2 * 3
    for repetition in (1, 2):
        for sample_id in ("a", "b", "c", "d"):
            observed = {
                item.strategy
                for item in first
                if item.repetition == repetition and item.sample_id == sample_id
            }
            assert observed == {
                "jev",
                "openai",
                "hybrid",
            }


def test_preview_makes_no_provider_calls(tmp_path: Path) -> None:
    jev = FakeProvider("jev", SentimentLabel.negative, 0.9)
    openai = FakeProvider("openai", SentimentLabel.positive)
    result = run_live_benchmark(
        _records(),
        jev=jev,
        openai=openai,
        threshold=0.8,
        output_dir=tmp_path,
        safety=LiveSafety(live=False),
        sample_size=2,
        repetitions=1,
        seed=1,
    )

    assert result.complete is False
    assert result.measurement_scope == "preview"
    assert jev.calls == openai.calls == 0
    assert not (tmp_path / "attempts.jsonl").exists()


def test_live_records_attempt_and_operation_evidence(tmp_path: Path) -> None:
    jev = FakeProvider("jev", SentimentLabel.negative, 0.5)
    openai = FakeProvider("openai", SentimentLabel.positive)
    result = run_live_benchmark(
        _records().head(1),
        jev=jev,
        openai=openai,
        threshold=0.8,
        output_dir=tmp_path,
        safety=LiveSafety(live=True, max_requests=4, max_estimated_cost_usd=1),
        sample_size=1,
        repetitions=1,
        seed=1,
        cost_per_request_usd=0.1,
    )

    assert result.complete is True
    assert len(result.operations) == 3
    hybrid = next(item for item in result.operations if item.strategy == "hybrid")
    assert [attempt.stage for attempt in hybrid.attempts] == ["jev", "openai"]
    assert hybrid.operation_wall_ms >= sum(item.stage_wall_ms for item in hybrid.attempts)
    assert all(item.cache_status == "bypassed" for item in hybrid.attempts)
    assert all(item.attempt_number == 1 for item in hybrid.attempts)
    assert (tmp_path / "attempts.jsonl").is_file()
    assert (tmp_path / "operations.jsonl").is_file()


def test_operation_timing_excludes_incremental_ledger_writes(
    tmp_path: Path, monkeypatch
) -> None:
    from cx_sentiment import benchmark

    original_append = benchmark._append_jsonl

    def slow_append(path: Path, model) -> None:
        time.sleep(0.02)
        original_append(path, model)

    monkeypatch.setattr(benchmark, "_append_jsonl", slow_append)
    result = run_live_benchmark(
        _records().head(1),
        jev=FakeProvider("jev", SentimentLabel.negative, 0.9),
        openai=FakeProvider("openai", SentimentLabel.positive),
        threshold=0.8,
        output_dir=tmp_path,
        safety=LiveSafety(live=True, max_requests=3, max_estimated_cost_usd=1),
        sample_size=1,
        repetitions=1,
        seed=1,
        cost_per_request_usd=0.1,
    )

    assert max(item.operation_wall_ms for item in result.operations) < 10


def test_live_budget_stop_checkpoints_incomplete_run(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="request limit"):
        run_live_benchmark(
            _records().head(1),
            jev=FakeProvider("jev", SentimentLabel.negative, 0.5),
            openai=FakeProvider("openai", SentimentLabel.positive),
            threshold=0.8,
            output_dir=tmp_path,
            safety=LiveSafety(live=True, max_requests=1, max_estimated_cost_usd=1),
            sample_size=1,
            repetitions=1,
            seed=1,
            cost_per_request_usd=0.1,
        )
    assert (tmp_path / "benchmark.json").is_file()
