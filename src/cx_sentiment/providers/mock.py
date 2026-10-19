from __future__ import annotations

import hashlib

from cx_sentiment.models import Prediction, SentimentLabel


class MockProvider:
    provider_name = "mock"
    model_id = "deterministic-v1"
    prompt_version = "mock-v1"
    requires_network = False

    def predict(self, text: str, language: str, sample_id: str) -> Prediction:
        digest = hashlib.sha256(f"{language}\0{text}".encode()).digest()
        raw = [digest[0] + 1, digest[1] + 1, digest[2] + 1]
        total = float(sum(raw))
        scores = {label: value / total for label, value in zip(SentimentLabel, raw, strict=True)}
        predicted = max(scores, key=scores.__getitem__)
        return Prediction(
            provider=self.provider_name,
            model_id=self.model_id,
            returned_model_id=self.model_id,
            prompt_version=self.prompt_version,
            predicted_label=predicted,
            scores_by_class=scores,
            provider_reported_confidence=None,
            confidence_definition=None,
            latency_ms=0,
            usage_input_tokens=0,
            usage_output_tokens=0,
            reported_credits=0,
            cost_usd_estimate=0,
            sample_id=sample_id,
        )
