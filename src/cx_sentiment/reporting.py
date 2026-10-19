from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from typing import Any

from cx_sentiment.metrics import EvaluationMetrics
from cx_sentiment.models import Prediction
from cx_sentiment.routing import ThresholdSelection
from cx_sentiment.runner import RunAccounting


def write_report(
    output_dir: Path,
    predictions: Sequence[Prediction],
    metrics: EvaluationMetrics,
    *,
    title: str,
    metadata: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "predictions.jsonl", predictions)
    atomic_write(output_dir / "metrics.json", metrics.model_dump_json(indent=2) + "\n")
    _write_metrics_csv(output_dir / "metrics.csv", metrics)
    # Re-rendering predictions must preserve an existing run's incomplete marker.
    metadata_path = output_dir / "report_metadata.json"
    saved_metadata = {}
    if metadata_path.exists():
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["metadata"]
    combined_metadata = {**saved_metadata, **(metadata or {})}
    if saved_metadata.get("complete") is False:
        combined_metadata["complete"] = False
    write_metadata(output_dir, title, combined_metadata)
    source = output_dir / "report.qmd"
    source.write_text(
        files("cx_sentiment").joinpath("reports/benchmark.qmd").read_text(), encoding="utf-8"
    )
    render_quarto(source)


def atomic_write(path: Path, content: str) -> None:
    """Replace one artifact only after its complete contents have reached disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_predictions(path: Path, predictions: Sequence[Prediction]) -> None:
    atomic_write(path, "".join(prediction.model_dump_json() + "\n" for prediction in predictions))


def write_metadata(output_dir: Path, title: str, metadata: dict[str, object]) -> None:
    atomic_write(
        output_dir / "report_metadata.json",
        json.dumps({"title": title, "metadata": metadata}, ensure_ascii=False, indent=2) + "\n",
    )


def checkpoint_run(
    output_dir: Path,
    predictions: Sequence[Prediction],
    accounting: RunAccounting,
    *,
    title: str,
    metadata: dict[str, Any],
) -> None:
    metadata.update(
        complete=accounting.complete,
        stop_reason=accounting.stop_reason,
        returned_models=sorted(
            {
                prediction.returned_model_id
                for prediction in predictions
                if prediction.returned_model_id is not None
            }
        ),
        returned_model_unknown_records=sum(
            prediction.returned_model_id is None for prediction in predictions
        ),
    )
    # Completion metadata is last: an interrupted checkpoint remains conservative.
    write_predictions(output_dir / "predictions.jsonl", predictions)
    atomic_write(
        output_dir / "execution.json",
        json.dumps({"format_version": 1, **accounting.model_dump(mode="json")}, indent=2) + "\n",
    )
    write_metadata(output_dir, title, metadata)


def _write_metrics_csv(path: Path, metrics: EvaluationMetrics) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scope",
                "support",
                "accuracy",
                "macro_f1",
                "coverage",
                "latency_p50_ms",
                "latency_p95_ms",
                "cost_usd_total",
                "cost_usd_per_1000",
                "cost_priced_fraction",
                "reported_credits_total",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scope": "overall",
                "support": metrics.support,
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "coverage": metrics.operational_coverage,
                "latency_p50_ms": metrics.latency_p50_ms,
                "latency_p95_ms": metrics.latency_p95_ms,
                "cost_usd_total": metrics.cost_usd_total,
                "cost_usd_per_1000": metrics.cost_usd_per_1000,
                "cost_priced_fraction": metrics.cost_priced_fraction,
                "reported_credits_total": metrics.reported_credits_total,
            }
        )
        for language, values in metrics.by_language.items():
            writer.writerow({"scope": language, **values, "coverage": ""})


class ReportError(RuntimeError):
    """Saved report data could not be rendered with Quarto."""


def require_quarto() -> str:
    executable = shutil.which("quarto")
    if executable is None:
        raise ReportError(
            "Quarto is required to render reports. Install the Quarto CLI from "
            "https://quarto.org/docs/get-started/ and run uv sync. "
            "See docs/REPORTING.md for setup."
        )
    return executable


def render_quarto(source: Path) -> None:
    executable = require_quarto()
    try:
        subprocess.run(
            [executable, "render", source.name, "--to", "html"],
            cwd=source.parent.resolve(),
            env={**os.environ, "QUARTO_PYTHON": sys.executable},
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReportError(
            f"Quarto could not render {source}. Saved data is intact. "
            f"Run quarto render '{source}' --to html to inspect the rendering error."
        ) from exc


def write_selection_report(path: Path, selection: ThresholdSelection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(selection), indent=2) + "\n", encoding="utf-8")
    source = path.with_suffix(".qmd")
    source.write_text(
        files("cx_sentiment").joinpath("reports/selection.qmd").read_text(), encoding="utf-8"
    )
    render_quarto(source)


def write_comparison_report(output_dir: Path, comparison: dict[str, Any]) -> None:
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source = output_dir / "report.qmd"
    source.write_text(
        files("cx_sentiment").joinpath("reports/comparison.qmd").read_text(), encoding="utf-8"
    )
    render_quarto(source)
