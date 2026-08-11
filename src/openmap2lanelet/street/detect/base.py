"""Object detection interface for street-level frames."""

from __future__ import annotations

from typing import Protocol

from ..types import Detection


class DetectorBackend(Protocol):
    name: str

    def detect(self, frames) -> list[Detection]:
        """Detect regulatory objects in posed frames."""
        ...
