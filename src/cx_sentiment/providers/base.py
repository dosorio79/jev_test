from __future__ import annotations

from typing import Protocol

from cx_sentiment.models import Prediction


class ProviderError(RuntimeError):
    """A classified provider-boundary failure."""

    def __init__(self, message: str, *, kind: str = "provider_error") -> None:
        super().__init__(message)
        self.kind = kind


class SentimentProvider(Protocol):
    provider_name: str
    model_id: str
    prompt_version: str
    requires_network: bool

    def predict(self, text: str, language: str, sample_id: str) -> Prediction: ...
