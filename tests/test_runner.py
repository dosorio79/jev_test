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
