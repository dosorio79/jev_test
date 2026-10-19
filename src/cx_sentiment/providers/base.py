from __future__ import annotations

import hashlib
import math
from typing import Any, Protocol
from urllib.parse import urlsplit

from cx_sentiment.models import Prediction


class ProviderError(RuntimeError):
    """A classified provider-boundary failure."""

    def __init__(
        self, message: str, *, kind: str = "provider_error", prediction: Prediction | None = None
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.prediction = prediction


def valid_token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def valid_nonnegative_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return float(value)
    return None


def endpoint_settings(endpoint: str) -> dict[str, str]:
    """Record an endpoint identity without copying URL credentials, paths, or query secrets."""
    parsed = urlsplit(endpoint)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return {
        "endpoint_origin": f"{parsed.scheme}://{host}{port}",
        "endpoint_identity": "sha256:" + hashlib.sha256(endpoint.encode()).hexdigest(),
    }


def provider_settings(provider: SentimentProvider) -> dict[str, Any]:
    """Effective non-secret provider settings used for cache keys and run provenance."""
    return dict(getattr(provider, "effective_settings", {}))


class SentimentProvider(Protocol):
    provider_name: str
    model_id: str
    prompt_version: str
    requires_network: bool

    def predict(self, text: str, language: str, sample_id: str) -> Prediction: ...
