from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from openai.types.shared import ReasoningEffort
from pydantic import ValidationError

from cx_sentiment.benchmark import run_live_benchmark
from cx_sentiment.budget import BudgetExceeded, LiveSafety, PriceCard
from cx_sentiment.cache import PredictionCache, RawResponseStore
from cx_sentiment.comparison import (
    compare_runs,
    create_comparison_directory,
    load_comparison_config,
)
from cx_sentiment.config import create_run_directory, load_config
from cx_sentiment.data import (
    DataValidationError,
    audit_dataset,
    create_validation_sample,
    load_dataset,
)
from cx_sentiment.metrics import evaluate_predictions
from cx_sentiment.models import Prediction
from cx_sentiment.pricing import get_jev_price_card, get_openai_price_card
from cx_sentiment.provenance import run_metadata
from cx_sentiment.providers.base import SentimentProvider
from cx_sentiment.providers.jev import DEFAULT_JEV_BASE_URL, DEFAULT_JEV_MODEL, JevProvider
from cx_sentiment.providers.mock import MockProvider
from cx_sentiment.providers.openai_provider import OpenAIProvider
from cx_sentiment.reporting import (
    ReportError,
    checkpoint_run,
    require_quarto,
    write_comparison_report,
    write_report,
    write_selection_report,
)
from cx_sentiment.routing import select_threshold
from cx_sentiment.runner import RunAccounting, run_predictions
from cx_sentiment.study import _read_saved_run, analyze_saved_validation
from cx_sentiment.study_config import create_study_directory, load_study_config


def _records(frame: pl.DataFrame) -> list[tuple[str, str, str]]:
    return list(frame.select("sample_id", "language", "text").iter_rows())


def _load_predictions(path: Path) -> list[Prediction]:
    return [
        Prediction.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_run(
    frame: pl.DataFrame,
    provider: SentimentProvider,
    output_dir: Path,
    safety: LiveSafety,
    *,
    max_cost_per_request_usd: float = 0,
) -> None:
    require_quarto()
    title = "Offline benchmark" if provider.provider_name == "mock" else "Live smoke test"
    metadata = run_metadata(frame, provider)
    metadata["execution_limits"] = {
        **safety.model_dump(),
        "cost_per_request_usd": max_cost_per_request_usd,
    }
    accounting = RunAccounting()

    def save_progress(predictions: list[Prediction], state: RunAccounting) -> None:
        checkpoint_run(output_dir, predictions, state, title=title, metadata=metadata)

    reasoning_effort = getattr(provider, "reasoning_effort", None)
    predictions = run_predictions(
        provider,
        _records(frame),
        cache=PredictionCache(Path(".cache/cx_sentiment")),
        safety=safety,
        settings={"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {},
        next_request_upper_bound_usd=max_cost_per_request_usd,
        accounting=accounting,
        on_progress=save_progress,
    )
    metrics = evaluate_predictions(
        frame["sentiment"].to_list(),
        predictions,
        languages=frame["language"].to_list(),
    )
    write_report(
        output_dir,
        predictions,
        metrics,
        title=title,
        metadata=metadata,
    )
    print(f"Report: {output_dir / 'report.html'}")


def _provider_from_environment(
    name: str, *, reasoning_effort: ReasoningEffort = None
) -> SentimentProvider:
    raw_store = RawResponseStore(Path(".cache/cx_sentiment/raw"))
    if name == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required")
        model = os.getenv("OPENAI_MODEL")
        if not model:
            raise ValueError("OPENAI_MODEL is required")
        return OpenAIProvider(
            model=model,
            price_card=_price_card("openai", model),
            raw_store=raw_store,
            reasoning_effort=reasoning_effort,
        )
    api_key = os.getenv("JEV_API_KEY")
    if not api_key:
        raise ValueError("JEV_API_KEY is required")
    model = os.getenv("JEV_MODEL", DEFAULT_JEV_MODEL)
    return JevProvider(
        api_key=api_key,
        base_url=os.getenv("JEV_BASE_URL", DEFAULT_JEV_BASE_URL),
        model=model,
        raw_store=raw_store,
        price_card=_price_card("jev", model),
    )


def _price_card(provider: str, model: str) -> PriceCard:
    try:
        return get_openai_price_card(model) if provider == "openai" else get_jev_price_card(model)
    except KeyError:
        raise ValueError(
            f"No verified {provider} rate card; select a supported model or add a dated rate card "
            "in pricing.py."
        ) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cx-sentiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare", help="Compare saved runs without provider calls")
    compare.add_argument("config", type=Path, help="Comparison YAML file")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--split", choices=["train", "validation", "test"])
    sample = subparsers.add_parser("sample-validation")
    sample.add_argument("--archive", type=Path, required=True)
    sample.add_argument("--output-dir", type=Path, default=Path("data"))
    selection = subparsers.add_parser("select-threshold")
    selection.add_argument("--input", type=Path, required=True)
    selection.add_argument("--decision-predictions", type=Path, required=True)
    selection.add_argument("--fallback-predictions", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("--max-f1-degradation", type=float, default=0.02)
    selection.add_argument("--decision-cost-usd", type=float)
    selection.add_argument("--fallback-cost-usd", type=float)
    report = subparsers.add_parser("report")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--predictions", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--title", default="CX sentiment benchmark")
    demo = subparsers.add_parser("offline-demo")
    demo.add_argument("--input", type=Path, required=True)
    demo.add_argument("--output-dir", type=Path, required=True)
    demo.add_argument("--limit", type=int, default=5)
    run = subparsers.add_parser("run", help="Run or preview a YAML experiment")
    run.add_argument("config", type=Path, nargs="?", help="Experiment YAML file")
    run.add_argument("--live", action="store_true", help="Allow paid provider calls")
    # Existing scripts remain usable; YAML is the documented interface.
    run.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    run.add_argument("--provider", choices=["openai", "jev"], help=argparse.SUPPRESS)
    run.add_argument("--max-requests", type=int, help=argparse.SUPPRESS)
    run.add_argument("--max-estimated-cost-usd", type=float, help=argparse.SUPPRESS)
    run.add_argument("--max-cost-per-request-usd", type=float, help=argparse.SUPPRESS)
    study = subparsers.add_parser("study", help="Analyze or preview a routing study")
    study.add_argument("config", type=Path, help="Study YAML file")
    study.add_argument("--live", action="store_true", help="Allow paid benchmark calls")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            _apply_run_config(args, parser)
        return _execute(args)
    except ReportError as exc:
        parser.error(str(exc))
    except DataValidationError:
        parser.error("invalid or unreadable dataset; check its columns, values, and selected split")
    except BudgetExceeded:
        parser.error(
            f"execution stopped at a request or USD budget limit; incomplete results saved in "
            f"{args.output_dir}"
        )
    except KeyboardInterrupt:
        print("Execution interrupted; completed results remain in the run folder.")
        return 130
    except ValueError as exc:
        # Pydantic errors can contain input values; keep only field locations.
        if isinstance(exc, ValidationError):
            fields = ", ".join(".".join(map(str, e["loc"])) or "settings" for e in exc.errors())
            parser.error(f"invalid settings: {fields}")
        parser.error(str(exc))
    except OSError:
        parser.error("cannot read or write experiment files; check paths and permissions")


def _execute(args: argparse.Namespace) -> int:
    if args.command == "study":
        config = load_study_config(args.config)
        output_dir = create_study_directory(config)
        if config.mode == "offline":
            result = analyze_saved_validation(config)
            (output_dir / "study.json").write_text(
                result.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            print(f"Study: {output_dir / 'study.json'} (no provider calls)")
            return 0
        assert config.live is not None
        maximum_requests = config.live.sample_size * config.live.repetitions * 4
        if not args.live:
            preview = {
                "measurement_scope": "preview",
                "sample_size": config.live.sample_size,
                "repetitions": config.live.repetitions,
                "planned_operations": config.live.sample_size
                * config.live.repetitions
                * 3,
                "maximum_provider_requests": maximum_requests,
                "configured_request_cap": config.live.max_requests,
                "configured_budget_usd": config.live.budget_usd,
                "network_calls": 0,
            }
            (output_dir / "preview.json").write_text(
                json.dumps(preview, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Preview: {output_dir / 'preview.json'} (no provider calls)")
            return 0
        if config.live.concurrency != 1:
            raise ValueError("the first authorized live benchmark requires concurrency: 1")
        records, _ = _read_saved_run(config.inputs.jev_run, "jev")
        result = run_live_benchmark(
            records,
            jev=_provider_from_environment("jev"),
            openai=_provider_from_environment("openai"),
            threshold=config.live.routing_threshold,
            output_dir=output_dir,
            safety=LiveSafety(
                live=True,
                max_requests=config.live.max_requests,
                max_estimated_cost_usd=config.live.budget_usd,
            ),
            sample_size=config.live.sample_size,
            repetitions=config.live.repetitions,
            seed=config.seed,
            cost_per_request_usd=config.live.cost_per_request_usd,
        )
        print(f"Benchmark: {output_dir / 'benchmark.json'} ({result.requests_made} calls)")
        return 0
    if args.command == "compare":
        config = load_comparison_config(args.config)
        comparison = compare_runs(config)
        require_quarto()
        output_dir = create_comparison_directory(config)
        write_comparison_report(output_dir, comparison)
        print(f"Report: {output_dir / 'report.html'} (no provider calls)")
        return 0
    if args.command == "audit":
        audit = audit_dataset(load_dataset(args.input, expected_split=args.split))
        print(json.dumps(audit.__dict__, indent=2))
        return 0
    if args.command == "sample-validation":
        sample = create_validation_sample(args.archive, args.output_dir)
        print(f"Created {sample.frame.height} validation records")
        return 0
    if args.command == "select-threshold":
        frame = load_dataset(args.input, expected_split="validation")
        selected = select_threshold(
            frame,
            _load_predictions(args.decision_predictions),
            _load_predictions(args.fallback_predictions),
            max_f1_degradation=args.max_f1_degradation,
            decision_cost=args.decision_cost_usd,
            fallback_cost=args.fallback_cost_usd,
        )
        write_selection_report(args.output, selected)
        return 0
    if args.command == "report":
        frame = load_dataset(args.input)
        predictions = _load_predictions(args.predictions)
        metrics = evaluate_predictions(
            frame["sentiment"].to_list(),
            predictions,
            languages=frame["language"].to_list(),
        )
        write_report(args.output_dir, predictions, metrics, title=args.title)
        return 0
    frame = load_dataset(args.input).head(args.limit)
    if frame.is_empty():
        raise DataValidationError("selected dataset is empty")
    if args.command == "offline-demo" or args.provider == "mock":
        _write_run(frame, MockProvider(), args.output_dir, LiveSafety())
        return 0
    if not args.live:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plan = {
            "provider": args.provider,
            "would_process": frame.height,
            "network_calls": 0,
            "reason": "dry run; add --live plus request and cost caps to execute",
        }
        (args.output_dir / "dry_run.json").write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Preview: {args.output_dir / 'dry_run.json'} (no network calls)")
        return 0
    if args.max_cost_per_request_usd is None or args.max_cost_per_request_usd <= 0:
        raise ValueError("--max-cost-per-request-usd is required and must be positive")
    safety = LiveSafety(
        live=True,
        max_requests=args.max_requests,
        max_estimated_cost_usd=args.max_estimated_cost_usd,
    )
    provider = _provider_from_environment(
        args.provider, reasoning_effort=getattr(args, "reasoning_effort", None)
    )
    _write_run(
        frame,
        provider,
        args.output_dir,
        safety,
        max_cost_per_request_usd=args.max_cost_per_request_usd,
    )
    return 0


def _apply_run_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.config is None:
        if args.input is None or args.output_dir is None or args.provider is None:
            parser.error("run requires an experiment YAML file")
        if args.limit is None:
            args.limit = 5
        return
    legacy_options = (
        args.input,
        args.output_dir,
        args.limit,
        args.provider,
        args.max_requests,
        args.max_estimated_cost_usd,
        args.max_cost_per_request_usd,
    )
    if any(value is not None for value in legacy_options):
        parser.error("use a YAML file or legacy flags, not both")
    try:
        config = load_config(args.config)
    except ValueError as exc:
        parser.error(str(exc))
    if args.live and config.provider != "mock":
        if config.cost_per_request_usd is None:
            parser.error("set cost_per_request_usd in the experiment YAML for live calls")
        if config.budget_usd is None:
            parser.error("set budget_usd in the experiment YAML for live calls")
    args.input = config.input
    args.output_dir = create_run_directory(config, live=args.live)
    args.provider = config.provider
    args.limit = config.limit
    args.max_requests = config.limit
    args.max_estimated_cost_usd = config.budget_usd
    args.max_cost_per_request_usd = config.cost_per_request_usd
    args.reasoning_effort = config.reasoning_effort


if __name__ == "__main__":
    raise SystemExit(main())
