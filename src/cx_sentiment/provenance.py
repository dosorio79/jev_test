"""Small, non-secret provenance records for saved experiment artifacts."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import polars as pl

from cx_sentiment.comparison import dataset_fingerprint
from cx_sentiment.providers.base import SentimentProvider, provider_settings


def code_version() -> dict[str, object]:
    try:
        package_version = version("cx-sentiment-benchmark")
    except PackageNotFoundError:
        package_version = "unavailable"
    result: dict[str, object] = {"package_version": package_version}
    source_root = Path(__file__).resolve().parents[2]
    # Only use this checkout's Git metadata, never an unrelated invocation directory.
    if not (source_root / ".git").exists():
        return result
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return result
    result.update(git_commit=commit, git_dirty=bool(dirty))
    return result


def run_metadata(frame: pl.DataFrame, provider: SentimentProvider) -> dict[str, Any]:
    settings = provider_settings(provider)
    return {
        "format_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        **code_version(),
        "provider": provider.provider_name,
        "model": provider.model_id,
        "requested_model": provider.model_id,
        "returned_models": [],
        "returned_model_unknown_records": 0,
        "prompt_version": provider.prompt_version,
        "simulated": provider.provider_name == "mock",
        "reasoning_effort": getattr(provider, "reasoning_effort", None),
        "effective_settings": settings,
        "pricing": settings.get("price_card"),
        "dataset_sha256": dataset_fingerprint(frame),
        "selected_records": frame.height,
        "complete": False,
        "cost_and_usage_scope": (
            "Historical prediction estimates and usage, including original cached results. "
            "Invocation-only request counts and spend are recorded separately in execution.json."
        ),
        "latency_scope": "Recorded request timings, including original cached timings.",
        "collection_policy": "One attempt per uncached record; no automatic retries.",
    }
