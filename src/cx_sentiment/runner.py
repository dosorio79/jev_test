from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from cx_sentiment.budget import LiveSafety
from cx_sentiment.cache import PredictionCache
from cx_sentiment.models import Prediction
from cx_sentiment.providers.base import ProviderError, SentimentProvider


def run_predictions(
    provider: SentimentProvider,
    records: Iterable[tuple[str, str, str]],
    *,
    cache: PredictionCache,
    safety: LiveSafety,
    settings: dict[str, Any] | None = None,
    next_request_upper_bound_usd: float = 0,
) -> list[Prediction]:
    settings = settings or {}
    results: list[Prediction] = []
    requests_made = 0
    estimated_spend = 0.0
    for sample_id, language, text in records:
        key = cache.key(
            provider=provider.provider_name,
            model_id=provider.model_id,
            prompt_version=provider.prompt_version,
            text=text,
            settings={**settings, "language": language},
        )
        cached = cache.get(key)
        if cached is not None:
            results.append(cached.model_copy(update={"sample_id": sample_id}))
            continue
        if getattr(provider, "requires_network", True):
            safety.assert_can_call(requests_made, estimated_spend, next_request_upper_bound_usd)
        try:
            prediction = provider.predict(text, language, sample_id)
        except ProviderError as exc:
            prediction = Prediction(
                provider=provider.provider_name,
                model_id=provider.model_id,
                prompt_version=provider.prompt_version,
                predicted_label=None,
                latency_ms=0,
                error_kind=exc.kind,
                sample_id=sample_id,
            )
        requests_made += 1
        estimated_spend += max(prediction.cost_usd_estimate or 0, next_request_upper_bound_usd)
        cache.put(key, prediction)
        results.append(prediction)
    return results
