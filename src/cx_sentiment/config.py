from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ExperimentConfig(BaseModel):
    """One experiment; paths are relative to its YAML file."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    provider: Literal["mock", "openai", "jev"] = "mock"
    input: Path = Path("data/cx_sentiment_test_600.csv")
    output_dir: Path = Path("outputs/experiment")
    limit: int = Field(default=5, gt=0, strict=True)
    budget_usd: float | None = Field(default=None, gt=0)
    cost_per_request_usd: float | None = Field(default=None, gt=0)


def load_config(path: Path) -> ExperimentConfig:
    try:
        with path.open(encoding="utf-8") as handle:
            config = ExperimentConfig.model_validate(yaml.safe_load(handle))
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
    return config.model_copy(
        update={
            "input": path.parent / config.input,
            "output_dir": path.parent / config.output_dir,
        }
    )
