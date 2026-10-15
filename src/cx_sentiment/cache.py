from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cx_sentiment.models import Prediction


class PredictionCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def key(
        self,
        *,
        provider: str,
        model_id: str,
        prompt_version: str,
        text: str,
        settings: dict[str, Any],
    ) -> str:
        payload = {
            "provider": provider,
            "model_id": model_id,
            "prompt_version": prompt_version,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "settings": settings,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def get(self, key: str) -> Prediction | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        return Prediction.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, prediction: Prediction) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(prediction.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)


class RawResponseStore:
    """Private content-addressed storage for full provider responses."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, provider: str, sample_id: str, payload: bytes) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        digest = hashlib.sha256(payload).hexdigest()
        safe_sample = "".join(char for char in sample_id if char.isalnum() or char in "-_")[:40]
        path = self.directory / f"{provider}-{safe_sample}-{digest}.json"
        if not path.exists():
            path.write_bytes(payload)
            os.chmod(path, 0o600)
        return path
