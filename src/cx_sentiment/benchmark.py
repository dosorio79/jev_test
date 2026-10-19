from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from cx_sentiment.budget import LiveSafety
from cx_sentiment.models import Prediction
from cx_sentiment.providers.base import ProviderError, SentimentProvider

Strategy = Literal["jev", "openai", "hybrid"]


class ScheduledOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_index: int
    sample_id: str
    repetition: int
    strategy: Strategy
    condition: Literal["cold", "warm"] = "warm"


class BenchmarkAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    strategy: Strategy
    stage: Literal["jev", "openai"]
    repetition: int
    condition: Literal["cold", "warm"]
    attempt_number: int = 1
    order_index: int
    stage_wall_ms: float = Field(ge=0)
    retry_backoff_ms: float = 0
    outcome: Literal["success", "failure"]
    usage_input_tokens: int | None = None
    usage_output_tokens: int | None = None
    cost_usd: float | None = None
    cache_status: Literal["bypassed", "hit", "miss"] = "bypassed"
    error_kind: str | None = None


class BenchmarkOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    strategy: Strategy
    repetition: int
    condition: Literal["cold", "warm"]
    order_index: int
    scheduler_wait_ms: float = Field(ge=0)
    operation_wall_ms: float = Field(ge=0)
    outcome: Literal["success", "failure"]
    predicted_label: str | None
    attempts: tuple[BenchmarkAttempt, ...]


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    format_version: int = 1
    measurement_scope: Literal["preview", "controlled"]
    complete: bool
    planned_operations: int
    requests_made: int
    estimated_spend_usd: float
    operations: tuple[BenchmarkOperation, ...] = ()
    stop_reason: str | None = None


def _stratified_sample(records: pl.DataFrame, sample_size: int, seed: int) -> pl.DataFrame:
    if sample_size > records.height:
        raise ValueError("sample_size exceeds available records")
    shuffled = records.sample(fraction=1, shuffle=True, seed=seed)
    strata = shuffled.partition_by(["language", "sentiment"], maintain_order=True)
    selected: list[dict[str, object]] = []
    offsets = [0] * len(strata)
    while len(selected) < sample_size:
        progressed = False
        for index, stratum in enumerate(strata):
            if offsets[index] < stratum.height and len(selected) < sample_size:
                selected.append(stratum.row(offsets[index], named=True))
                offsets[index] += 1
                progressed = True
        if not progressed:
            break
    return pl.DataFrame(selected, schema=records.schema)


def build_schedule(
    records: pl.DataFrame, *, sample_size: int, repetitions: int, seed: int
) -> tuple[ScheduledOperation, ...]:
    selected = _stratified_sample(records, sample_size, seed)
    rng = random.Random(seed)
    pending: list[tuple[str, int, Strategy]] = []
    for repetition in range(1, repetitions + 1):
        ids = selected["sample_id"].to_list()
        rng.shuffle(ids)
        for position, sample_id in enumerate(ids):
            strategies: list[Strategy] = ["jev", "openai", "hybrid"]
            rotation = (position + repetition - 1) % len(strategies)
            strategies = strategies[rotation:] + strategies[:rotation]
            pending.extend((sample_id, repetition, strategy) for strategy in strategies)
    return tuple(
        ScheduledOperation(
            order_index=index,
            sample_id=sample_id,
            repetition=repetition,
            strategy=strategy,
        )
        for index, (sample_id, repetition, strategy) in enumerate(pending)
    )


def _append_jsonl(path: Path, model: BaseModel) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(model.model_dump_json() + "\n")


def run_live_benchmark(
    records: pl.DataFrame,
    *,
    jev: SentimentProvider,
    openai: SentimentProvider,
    threshold: float,
    output_dir: Path,
    safety: LiveSafety,
    sample_size: int,
    repetitions: int,
    seed: int,
    cost_per_request_usd: float = 0,
) -> BenchmarkRun:
    schedule = build_schedule(
        records, sample_size=sample_size, repetitions=repetitions, seed=seed
    )
    if not safety.live:
        return BenchmarkRun(
            measurement_scope="preview",
            complete=False,
            planned_operations=len(schedule),
            requests_made=0,
            estimated_spend_usd=0,
            stop_reason="network calls require explicit --live",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {row["sample_id"]: row for row in records.iter_rows(named=True)}
    operations: list[BenchmarkOperation] = []
    requests_made = 0
    spend = 0.0

    def call(
        provider: SentimentProvider, item: ScheduledOperation, stage: Literal["jev", "openai"]
    ) -> tuple[Prediction | None, BenchmarkAttempt]:
        nonlocal requests_made, spend
        safety.assert_can_call(requests_made, spend, cost_per_request_usd)
        row = rows[item.sample_id]
        started = time.perf_counter()
        prediction: Prediction | None = None
        error_kind: str | None = None
        try:
            prediction = provider.predict(str(row["text"]), str(row["language"]), item.sample_id)
            outcome: Literal["success", "failure"] = (
                "success" if prediction.predicted_label is not None else "failure"
            )
            error_kind = prediction.error_kind
        except ProviderError as exc:
            prediction = exc.prediction
            outcome = "failure"
            error_kind = exc.kind
        elapsed = (time.perf_counter() - started) * 1000
        requests_made += 1
        actual_cost = prediction.cost_usd_estimate if prediction is not None else None
        spend += actual_cost if actual_cost is not None else cost_per_request_usd
        attempt = BenchmarkAttempt(
            sample_id=item.sample_id,
            strategy=item.strategy,
            stage=stage,
            repetition=item.repetition,
            condition=item.condition,
            order_index=item.order_index,
            stage_wall_ms=elapsed,
            outcome=outcome,
            usage_input_tokens=(prediction.usage_input_tokens if prediction else None),
            usage_output_tokens=(prediction.usage_output_tokens if prediction else None),
            cost_usd=actual_cost,
            error_kind=error_kind,
        )
        _append_jsonl(output_dir / "attempts.jsonl", attempt)
        return prediction, attempt

    try:
        for item in schedule:
            operation_started = time.perf_counter()
            attempts: list[BenchmarkAttempt] = []
            prediction: Prediction | None
            if item.strategy == "jev":
                prediction, attempt = call(jev, item, "jev")
                attempts.append(attempt)
            elif item.strategy == "openai":
                prediction, attempt = call(openai, item, "openai")
                attempts.append(attempt)
            else:
                prediction, attempt = call(jev, item, "jev")
                attempts.append(attempt)
                confidence = (
                    max(prediction.scores_by_class.values())
                    if prediction is not None and prediction.scores_by_class is not None
                    else None
                )
                needs_fallback = (
                    prediction is None
                    or prediction.predicted_label is None
                    or confidence is None
                    or confidence < threshold
                )
                if needs_fallback:
                    prediction, attempt = call(openai, item, "openai")
                    attempts.append(attempt)
            operation = BenchmarkOperation(
                sample_id=item.sample_id,
                strategy=item.strategy,
                repetition=item.repetition,
                condition=item.condition,
                order_index=item.order_index,
                scheduler_wait_ms=0,
                operation_wall_ms=(time.perf_counter() - operation_started) * 1000,
                outcome=(
                    "success"
                    if prediction is not None and prediction.predicted_label is not None
                    else "failure"
                ),
                predicted_label=(
                    prediction.predicted_label.value
                    if prediction is not None and prediction.predicted_label is not None
                    else None
                ),
                attempts=tuple(attempts),
            )
            operations.append(operation)
            _append_jsonl(output_dir / "operations.jsonl", operation)
    except (Exception, KeyboardInterrupt) as exc:
        result = BenchmarkRun(
            measurement_scope="controlled",
            complete=False,
            planned_operations=len(schedule),
            requests_made=requests_made,
            estimated_spend_usd=spend,
            operations=tuple(operations),
            stop_reason=str(exc) or type(exc).__name__,
        )
        (output_dir / "benchmark.json").write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        raise
    result = BenchmarkRun(
        measurement_scope="controlled",
        complete=True,
        planned_operations=len(schedule),
        requests_made=requests_made,
        estimated_spend_usd=spend,
        operations=tuple(operations),
    )
    (output_dir / "benchmark.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return result
