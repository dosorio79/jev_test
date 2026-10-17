"""Compare saved runs without constructing providers or making network calls."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cx_sentiment.config import load_config
from cx_sentiment.data import load_dataset
from cx_sentiment.metrics import evaluate_predictions
from cx_sentiment.models import Prediction


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="CX sentiment comparison", min_length=1)
    runs: list[Path] = Field(min_length=2)
    output_dir: Path = Path("runs")


def load_comparison_config(path: Path) -> ComparisonConfig:
    try:
        config = ComparisonConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}") from exc
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, e["loc"])) or "configuration" for e in exc.errors())
        raise ValueError(f"invalid comparison settings in {path}: {fields}") from exc
    runs = [(path.parent / run).resolve() for run in config.runs]
    if len(set(runs)) != len(runs):
        raise ValueError(f"comparison requires distinct run folders in {path}")
    return config.model_copy(
        update={
            "runs": runs,
            "output_dir": (path.parent / config.output_dir).resolve(),
        }
    )


def dataset_fingerprint(frame: pl.DataFrame) -> str:
    """Hash selected source records, including text, labels, and provenance."""
    canonical = frame.sort("sample_id").write_csv()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_comparison_directory(config: ComparisonConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    output = config.output_dir / f"{timestamp}-comparison"
    output.mkdir(parents=True)
    (output / "comparison.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return output


def compare_runs(config: ComparisonConfig) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    warnings: list[str] = []
    reference: pl.DataFrame | None = None
    reference_hash: str | None = None
    labels_by_run: dict[str, dict[str, str | None]] = {}
    for index, directory in enumerate(config.runs, start=1):
        experiment = load_config(directory / "experiment.yaml")
        frame = load_dataset(experiment.input).head(experiment.limit).sort("sample_id")
        if frame.is_empty():
            raise ValueError(f"no source records in {directory}")
        fingerprint = dataset_fingerprint(frame)
        if reference is None:
            reference, reference_hash = frame, fingerprint
        elif fingerprint != reference_hash:
            raise ValueError(f"runs do not contain identical source records: {directory}")
        try:
            predictions = [
                Prediction.model_validate_json(line)
                for line in (directory / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            report = json.loads((directory / "report_metadata.json").read_text(encoding="utf-8"))
            metadata = report["metadata"]
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            title = report.get("title", directory.name)
            if not isinstance(title, str):
                raise ValueError("title must be text")
            for field in ("cost_and_usage_scope", "latency_scope", "collection_policy"):
                if field in metadata and not isinstance(metadata[field], str):
                    raise ValueError(f"{field} must be text")
            recollection_attempts = None
            recollection_cost = None
            audit_path = directory / "recollection_audit.json"
            if audit_path.exists():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if not isinstance(audit, list):
                    raise ValueError("recollection audit must be a list")
                costs = [attempt["fresh_cost_usd"] for attempt in audit]
                for cost in costs:
                    if cost is not None and (
                        type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
                    ):
                        raise ValueError("invalid recollection cost")
                recollection_attempts = len(costs)
                if all(cost is not None for cost in costs):
                    recollection_cost = float(sum(costs))
        except OSError as exc:
            raise ValueError(f"cannot read saved run {directory}: {exc.strerror}") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid saved predictions or metadata in {directory}") from exc
        ids = [prediction.sample_id for prediction in predictions]
        if len(set(ids)) != len(ids) or set(ids) != set(frame["sample_id"].to_list()):
            raise ValueError(
                f"prediction sample IDs do not match selected source records: {directory}"
            )
        identities = {(p.provider, p.model_id, p.prompt_version) for p in predictions}
        if len(identities) != 1:
            raise ValueError(f"mixed providers, models, or prompts in {directory}")
        historical_hash = metadata.get("dataset_sha256")
        if historical_hash is None:
            warnings.append(
                f"{directory.name}: historical input unverified; this legacy run did not save "
                "a dataset fingerprint. Metrics use the source data currently at its saved path."
            )
        elif historical_hash != fingerprint:
            raise ValueError(f"saved dataset fingerprint does not match current input: {directory}")
        by_id = {p.sample_id: p for p in predictions}
        aligned = [by_id[sample_id] for sample_id in frame["sample_id"].to_list()]
        metrics = evaluate_predictions(
            frame["sentiment"].to_list(),
            aligned,
            languages=frame["language"].to_list(),
        )
        label = f"Run {index}"
        provider, model, prompt = next(iter(identities))
        summaries.append(
            {
                "label": label,
                "title": title,
                "path": str(directory),
                "provider": provider,
                "model": model,
                "prompt_version": prompt,
                "simulated": provider == "mock",
                "historical_input_verified": historical_hash is not None,
                "cost_and_usage_scope": metadata.get(
                    "cost_and_usage_scope",
                    "Recorded costs and usage; cached results retain original estimates.",
                ),
                "latency_scope": metadata.get(
                    "latency_scope", "Recorded request times, including original cached timings."
                ),
                "collection_policy": metadata.get("collection_policy", "Not recorded"),
                "recollection_attempts": recollection_attempts,
                "recollection_cost_usd": recollection_cost,
                "metrics": metrics.model_dump(mode="json"),
            }
        )
        labels_by_run[label] = {p.sample_id: p.predicted_label for p in aligned}
    assert reference is not None
    disagreements = []
    for sample_id, language, truth in reference.select(
        "sample_id", "language", "sentiment"
    ).iter_rows():
        predicted = {label: values[sample_id] for label, values in labels_by_run.items()}
        if len(set(predicted.values())) > 1:
            disagreements.append(
                {
                    "sample_id": sample_id,
                    "language": language,
                    "truth": truth,
                    **{label: value or "unresolved" for label, value in predicted.items()},
                }
            )
    return {
        "title": config.title,
        "dataset_sha256": reference_hash,
        "historical_inputs_verified": not warnings,
        "warnings": warnings,
        "runs": summaries,
        "disagreements": disagreements,
    }
