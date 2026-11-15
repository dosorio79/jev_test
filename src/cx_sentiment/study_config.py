from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class OfflineInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    jev_run: Path
    openai_run: Path
    source_split: Literal["validation"] = "validation"


class LiveBenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    sample_size: int = Field(gt=0, strict=True)
    repetitions: int = Field(gt=0, strict=True)
    concurrency: Literal[1, 5, 10] = 1
    conditions: tuple[Literal["cold", "warm"], ...] = ("warm",)
    max_requests: int = Field(gt=0, strict=True)
    budget_usd: float = Field(gt=0)
    cost_per_request_usd: float = Field(gt=0)
    routing_threshold: float = Field(ge=0, le=1)


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    mode: Literal["offline", "live"] = "offline"
    inputs: OfflineInputs
    output_dir: Path = Path("runs")
    thresholds: tuple[int, ...] = tuple(range(40, 100))
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)
    max_macro_f1_delta: float = Field(default=0.01, ge=0, le=1)
    max_language_f1_delta: float = Field(default=0.03, ge=0, le=1)
    bootstrap_samples: int = Field(default=1000, gt=0, strict=True)
    seed: int = 20261020
    live: LiveBenchmarkConfig | None = None

    @model_validator(mode="after")
    def validate_study(self) -> Self:
        if not self.thresholds or any(point < 0 or point > 100 for point in self.thresholds):
            raise ValueError("thresholds must contain integer percentage points from 0 to 100")
        if len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError("thresholds must not contain duplicates")
        if not self.quantiles or any(value <= 0 or value >= 1 for value in self.quantiles):
            raise ValueError("quantiles must be between zero and one")
        if self.mode == "live" and self.live is None:
            raise ValueError("live settings are required when mode is live")
        if self.mode == "offline" and self.live is not None:
            raise ValueError("live settings require mode: live")
        return self


def load_study_config(path: Path) -> StudyConfig:
    try:
        with path.open(encoding="utf-8") as handle:
            config = StudyConfig.model_validate(yaml.safe_load(handle))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}") from exc
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'configuration'}: {error['msg']}"
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        )
        raise ValueError(f"invalid settings in {path}: {details}") from exc

    parent = path.parent
    inputs = config.inputs.model_copy(
        update={
            "jev_run": parent / config.inputs.jev_run,
            "openai_run": parent / config.inputs.openai_run,
        }
    )
    return config.model_copy(
        update={"inputs": inputs, "output_dir": parent / config.output_dir}
    )


def create_study_directory(config: StudyConfig) -> Path:
    root = config.output_dir.resolve()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    output = root / f"{timestamp}-routing-study-{config.mode}"
    output.mkdir(parents=True)
    snapshot = config.model_copy(update={"output_dir": root})
    (output / "study.yaml").write_text(
        yaml.safe_dump(snapshot.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return output
