from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from cx_sentiment.budget import BudgetExceeded, LiveSafety
from cx_sentiment.cache import PredictionCache
from cx_sentiment.models import Prediction
from cx_sentiment.providers.base import ProviderError, SentimentProvider, provider_settings


class RecordExecution(BaseModel):
    sample_id: str
    status: Literal["pending", "running", "cache_hit", "success", "failure", "skipped"] = "pending"
    attempted: bool = False
    network_request: bool = False
    error_kind: str | None = None
    cost_usd_estimate: float | None = None
    budget_reservation_usd: float = 0


class RunAccounting(BaseModel):
    ledger: list[RecordExecution] = Field(default_factory=list)
    request_count: int = 0
    known_spend_usd: float = 0
    unknown_cost_attempts: int = 0
    reserved_spend_usd: float = 0
    complete: bool = False
    stop_reason: str | None = None


ProgressCallback = Callable[[list[Prediction], RunAccounting], None]


def run_predictions(
    provider: SentimentProvider,
    records: Iterable[tuple[str, str, str]],
    *,
    cache: PredictionCache,
    safety: LiveSafety,
    settings: dict[str, Any] | None = None,
    next_request_upper_bound_usd: float = 0,
    accounting: RunAccounting | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[Prediction]:
    """Run sequentially; callbacks can persist results and invocation-only accounting.

    Budget stops and interruptions still raise, after the final progress callback.
    Cached predictions retain historical costs; they do not increase invocation spend.
    Reservations conservatively retain the greater of each bound and known cost.
    """
    if not math.isfinite(next_request_upper_bound_usd) or next_request_upper_bound_usd < 0:
        raise ValueError("request cost bound must be finite and nonnegative")
    selected = list(records)
    state = accounting if accounting is not None else RunAccounting()
    if state.ledger or state.request_count:
        raise ValueError("accounting must be empty for a new invocation")
    state.ledger = [RecordExecution(sample_id=sample_id) for sample_id, _, _ in selected]
    results: list[Prediction] = []
    active: RecordExecution | None = None

    def progress() -> None:
        if on_progress is not None:
            on_progress(results, state)

    try:
        progress()
        for entry, (sample_id, language, text) in zip(state.ledger, selected, strict=True):
            active = entry
            key = cache.key(
                provider=provider.provider_name,
                model_id=provider.model_id,
                prompt_version=provider.prompt_version,
                text=text,
                settings={
                    "provider_settings": provider_settings(provider),
                    "run_settings": settings or {},
                    "language": language,
                },
            )
            cached = cache.get(key)
            if cached is not None:
                prediction = cached.model_copy(update={"sample_id": sample_id})
                entry.status = "cache_hit"
                entry.error_kind = prediction.error_kind
            else:
                network = getattr(provider, "requires_network", True)
                if network:
                    safety.assert_can_call(
                        state.request_count, state.reserved_spend_usd, next_request_upper_bound_usd
                    )
                    state.request_count += 1
                    entry.network_request = True
                    entry.budget_reservation_usd = next_request_upper_bound_usd
                    state.reserved_spend_usd += next_request_upper_bound_usd
                entry.status = "running"
                entry.attempted = True
                started = time.perf_counter()
                try:
                    progress()
                except (Exception, KeyboardInterrupt):
                    # Persistence failed before predict was entered: no provider attempt.
                    if network:
                        state.request_count -= 1
                        state.reserved_spend_usd -= entry.budget_reservation_usd
                    entry.budget_reservation_usd = 0
                    entry.network_request = False
                    entry.attempted = False
                    entry.status = "pending"
                    raise
                try:
                    prediction = provider.predict(text, language, sample_id)
                except ProviderError as exc:
                    prediction = exc.prediction or Prediction(
                        provider=provider.provider_name,
                        model_id=provider.model_id,
                        prompt_version=provider.prompt_version,
                        predicted_label=None,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        error_kind=exc.kind,
                        sample_id=sample_id,
                    )
                entry.status = "failure" if prediction.error_kind is not None else "success"
                entry.error_kind = prediction.error_kind
                entry.cost_usd_estimate = prediction.cost_usd_estimate
                if network:
                    if prediction.cost_usd_estimate is None:
                        state.unknown_cost_attempts += 1
                    else:
                        state.known_spend_usd += prediction.cost_usd_estimate
                        extra = max(0, prediction.cost_usd_estimate - entry.budget_reservation_usd)
                        entry.budget_reservation_usd += extra
                        state.reserved_spend_usd += extra
                results.append(prediction)
                cache.put(key, prediction)
                progress()
                continue
            results.append(prediction)
            progress()
        state.complete = True
        progress()
    except (Exception, KeyboardInterrupt) as exc:
        state.complete = False
        state.stop_reason = (
            "budget_exceeded"
            if isinstance(exc, BudgetExceeded)
            else "interrupted"
            if isinstance(exc, KeyboardInterrupt)
            else "execution_error"
        )
        if active is not None and active.status == "running":
            active.status = "failure"
            active.error_kind = state.stop_reason
            if active.network_request:
                state.unknown_cost_attempts += 1
        for entry in state.ledger:
            if entry.status == "pending":
                entry.status = "skipped"
                entry.error_kind = state.stop_reason
        progress()
        raise
    return results
