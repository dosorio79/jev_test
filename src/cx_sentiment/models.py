from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SentimentLabel(StrEnum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"


LABEL_TO_ID = {
    SentimentLabel.negative: 0,
    SentimentLabel.neutral: 1,
    SentimentLabel.positive: 2,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


class Prediction(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    provider: str
    model_id: str
    returned_model_id: str | None = None
    prompt_version: str
    predicted_label: SentimentLabel | None
    scores_by_class: dict[SentimentLabel, float] | None = None
    provider_reported_confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_definition: str | None = None
    latency_ms: float = Field(ge=0)
    usage_input_tokens: int | None = Field(default=None, ge=0)
    usage_output_tokens: int | None = Field(default=None, ge=0)
    reported_credits: float | None = Field(default=None, ge=0)
    cost_usd_estimate: float | None = Field(default=None, ge=0)
    error_kind: str | None = None
    raw_response_path_or_hash: str | None = None
    sample_id: str

    @model_validator(mode="after")
    def validate_success_or_error(self) -> Prediction:
        if self.predicted_label is None and self.error_kind is None:
            raise ValueError("missing predictions require error_kind")
        if self.scores_by_class is not None:
            if set(self.scores_by_class) != set(SentimentLabel):
                raise ValueError("scores must contain all canonical classes")
            if any(not 0 <= score <= 1 for score in self.scores_by_class.values()):
                raise ValueError("scores must be in [0, 1]")
            if abs(sum(self.scores_by_class.values()) - 1.0) > 1e-4:
                raise ValueError("scores must sum to one")
        return self
