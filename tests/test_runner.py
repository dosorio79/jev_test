from __future__ import annotations

from pathlib import Path

import pytest

from cx_sentiment.budget import BudgetExceeded, LiveSafety, PriceCard
from cx_sentiment.cache import PredictionCache, RawResponseStore
from cx_sentiment.models import Prediction, SentimentLabel
from cx_sentiment.runner import run_predictions


class CountingProvider:
    provider_name = "counting"
    model_id = "v1"
    prompt_version = "p1"
    requires_network = True

    def __init__(self) -> None:
        self.calls = 0
        self.effective_settings: dict[str, str] = {}

    def predict(self, text: str, language: str, sample_id: str) -> Prediction:
        self.calls += 1
        return Prediction(
            provider=self.provider_name,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            predicted_label=SentimentLabel.neutral,
            latency_ms=1,
            usage_input_tokens=100,
            usage_output_tokens=10,
            cost_usd_estimate=0.001,
            sample_id=sample_id,
        )


def test_cache_key_includes_settings_and_avoids_second_call(tmp_path: Path) -> None:
    provider = CountingProvider()
    cache = PredictionCache(tmp_path / "cache")
    safety = LiveSafety(live=True, max_requests=5, max_estimated_cost_usd=1)

    first = run_predictions(
        provider,
        [("s1", "en", "hello")],
        cache=cache,
        safety=safety,
        settings={"temperature": 0},
    )
    second = run_predictions(
        provider,
        [("s1", "en", "hello")],
        cache=cache,
        safety=safety,
        settings={"temperature": 0},
    )

    assert first == second
    assert provider.calls == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_live_safety_requires_explicit_live_request_cap_and_budget() -> None:
    with pytest.raises(ValueError, match="--live"):
        LiveSafety(live=False, max_requests=5, max_estimated_cost_usd=1).assert_can_call(0, 0)
    with pytest.raises(ValueError, match="max_requests"):
        LiveSafety(live=True, max_requests=None, max_estimated_cost_usd=1)
    with pytest.raises(ValueError, match="budget"):
        LiveSafety(live=True, max_requests=5, max_estimated_cost_usd=None)


def test_budget_stops_before_request_that_can_exceed_cap() -> None:
    safety = LiveSafety(live=True, max_requests=5, max_estimated_cost_usd=0.001)
    with pytest.raises(BudgetExceeded):
        safety.assert_can_call(requests_made=0, estimated_spend_usd=0.0009, next_upper_bound=0.0002)


def test_runner_reserves_request_upper_bound_when_actual_cost_is_lower(tmp_path: Path) -> None:
    provider = CountingProvider()
    with pytest.raises(BudgetExceeded):
        run_predictions(
            provider,
            [(f"s{i}", "en", f"text {i}") for i in range(3)],
            cache=PredictionCache(tmp_path / "cache"),
            safety=LiveSafety(live=True, max_requests=5, max_estimated_cost_usd=0.025),
            next_request_upper_bound_usd=0.01,
        )

    assert provider.calls == 2


def test_price_card_uses_user_supplied_rates() -> None:
    card = PriceCard(input_usd_per_million=2, output_usd_per_million=8, as_of="2026-10-15")
    assert card.estimate(input_tokens=1000, output_tokens=100) == pytest.approx(0.0028)


def test_raw_response_store_is_private_and_content_addressed(tmp_path: Path) -> None:
    store = RawResponseStore(tmp_path / "raw")
    first = store.save("jev", "s1", b'{"answer":"positive"}')
    second = store.save("jev", "s1", b'{"answer":"positive"}')

    assert first == second
    assert first.read_bytes() == b'{"answer":"positive"}'
    assert first.stat().st_mode & 0o777 == 0o600


def test_accounting_separates_cached_costs_and_records_progress(tmp_path: Path) -> None:
    from cx_sentiment.runner import RunAccounting

    provider = CountingProvider()
    cache = PredictionCache(tmp_path / "cache")
    safety = LiveSafety(live=True, max_requests=5, max_estimated_cost_usd=1)
    run_predictions(provider, [("old", "en", "cached")], cache=cache, safety=safety)
    accounting = RunAccounting()
    snapshots = []
    results = run_predictions(
        provider,
        [("cached", "en", "cached"), ("fresh", "en", "fresh")],
        cache=cache,
        safety=safety,
        accounting=accounting,
        next_request_upper_bound_usd=0.01,
        on_progress=lambda results, state: snapshots.append(
            (len(results), state.model_dump(mode="json"))
        ),
    )
    assert len(results) == 2
    assert [entry.status for entry in accounting.ledger] == ["cache_hit", "success"]
    assert accounting.request_count == 1
    assert accounting.known_spend_usd == pytest.approx(0.001)
    assert accounting.reserved_spend_usd == pytest.approx(0.01)
    assert accounting.unknown_cost_attempts == 0
    assert accounting.complete
    assert snapshots[0][0] == 0
    assert [entry["status"] for entry in snapshots[0][1]["ledger"]] == ["pending", "pending"]
    assert snapshots[-1][1]["complete"]


def test_budget_stop_accounts_completed_results_and_skipped_records(tmp_path: Path) -> None:
    from cx_sentiment.runner import RunAccounting

    accounting = RunAccounting()
    snapshots = []
    with pytest.raises(BudgetExceeded):
        run_predictions(
            CountingProvider(),
            [(str(i), "en", str(i)) for i in range(3)],
            cache=PredictionCache(tmp_path / "cache"),
            safety=LiveSafety(live=True, max_requests=1, max_estimated_cost_usd=1),
            accounting=accounting,
            on_progress=lambda results, state: snapshots.append((len(results), state.complete)),
        )
    assert accounting.request_count == 1
    assert [entry.status for entry in accounting.ledger] == ["success", "skipped", "skipped"]
    assert accounting.stop_reason == "budget_exceeded"
    assert snapshots[-1] == (1, False)


def test_interruption_records_active_unknown_cost_and_remaining_skips(tmp_path: Path) -> None:
    from cx_sentiment.runner import RunAccounting

    class InterruptedProvider(CountingProvider):
        def predict(self, text: str, language: str, sample_id: str) -> Prediction:
            raise KeyboardInterrupt

    accounting = RunAccounting()
    with pytest.raises(KeyboardInterrupt):
        run_predictions(
            InterruptedProvider(),
            [("active", "en", "a"), ("next", "en", "b")],
            cache=PredictionCache(tmp_path / "cache"),
            safety=LiveSafety(live=True, max_requests=2, max_estimated_cost_usd=1),
            accounting=accounting,
            next_request_upper_bound_usd=0.1,
        )
    assert accounting.request_count == 1
    assert accounting.unknown_cost_attempts == 1
    assert accounting.known_spend_usd == 0
    assert accounting.reserved_spend_usd == 0.1
    assert [entry.status for entry in accounting.ledger] == ["failure", "skipped"]
    assert accounting.ledger[0].attempted
    assert accounting.ledger[0].error_kind == "interrupted"
    assert not accounting.ledger[1].attempted


def test_provider_errors_keep_measured_latency_and_unknown_cost(
    tmp_path: Path, monkeypatch
) -> None:
    from cx_sentiment.providers.base import ProviderError
    from cx_sentiment.runner import RunAccounting

    class FailedProvider(CountingProvider):
        def predict(self, text: str, language: str, sample_id: str) -> Prediction:
            raise ProviderError("redacted failure", kind="timeout")

    clock = iter([1.0, 1.25])
    monkeypatch.setattr("cx_sentiment.runner.time.perf_counter", lambda: next(clock))
    accounting = RunAccounting()
    results = run_predictions(
        FailedProvider(),
        [("failed", "en", "a")],
        cache=PredictionCache(tmp_path / "cache"),
        safety=LiveSafety(live=True, max_requests=1, max_estimated_cost_usd=1),
        accounting=accounting,
    )
    assert results[0].latency_ms == 250
    assert results[0].cost_usd_estimate is None
    assert accounting.unknown_cost_attempts == 1
    assert accounting.ledger[0].status == "failure"
    assert accounting.complete


def test_cache_separates_effective_provider_endpoints(tmp_path: Path) -> None:
    provider = CountingProvider()
    cache = PredictionCache(tmp_path / "cache")
    safety = LiveSafety(live=True, max_requests=5, max_estimated_cost_usd=1)
    provider.effective_settings = {"endpoint_identity": "first"}
    run_predictions(provider, [("s1", "en", "text")], cache=cache, safety=safety)
    provider.effective_settings = {"endpoint_identity": "second"}
    run_predictions(provider, [("s1", "en", "text")], cache=cache, safety=safety)
    assert provider.calls == 2


def test_mock_attempts_do_not_count_as_network_requests(tmp_path: Path) -> None:
    from cx_sentiment.providers.mock import MockProvider
    from cx_sentiment.runner import RunAccounting

    accounting = RunAccounting()
    run_predictions(
        MockProvider(),
        [("s1", "en", "text")],
        cache=PredictionCache(tmp_path / "cache"),
        safety=LiveSafety(),
        accounting=accounting,
    )
    assert accounting.complete
    assert accounting.ledger[0].attempted
    assert not accounting.ledger[0].network_request
    assert accounting.request_count == 0
    assert accounting.unknown_cost_attempts == 0


def test_callback_failure_before_provider_call_is_not_a_request(tmp_path: Path) -> None:
    from cx_sentiment.runner import RunAccounting

    provider = CountingProvider()
    accounting = RunAccounting()

    def progress(results, state):
        if state.ledger[0].status == "running":
            raise OSError("cannot persist")

    with pytest.raises(OSError, match="cannot persist"):
        run_predictions(
            provider,
            [("s1", "en", "text")],
            cache=PredictionCache(tmp_path / "cache"),
            safety=LiveSafety(live=True, max_requests=1, max_estimated_cost_usd=1),
            accounting=accounting,
            on_progress=progress,
            next_request_upper_bound_usd=0.1,
        )
    assert provider.calls == 0
    assert accounting.request_count == 0
    assert accounting.reserved_spend_usd == 0
    assert accounting.unknown_cost_attempts == 0
    assert accounting.ledger[0].status == "skipped"
    assert not accounting.ledger[0].attempted
