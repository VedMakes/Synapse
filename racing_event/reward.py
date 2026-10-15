"""Participant-facing declarative reward configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping


SIGNAL_NAMES = (
    "progress",
    "speed",
    "curvature_matched_speed",
    "off_track",
    "crash",
    "stuck",
    "lap_complete",
    "time",
    "steer_smoothness",
)


@dataclass(frozen=True)
class RewardConfig:
    weights: Mapping[str, float] = field(default_factory=lambda: {
        "progress": 12.0,
        "speed": 0.5,
        "curvature_matched_speed": 0.25,
        "off_track": -1.5,
        "crash": -10.0,
        "stuck": -8.0,
        "lap_complete": 20.0,
        "time": -0.15,
        "steer_smoothness": -0.05,
    })

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(SIGNAL_NAMES)
        if unknown:
            raise ValueError(f"unknown reward signals: {sorted(unknown)}")
        object.__setattr__(self, "weights", {name: float(self.weights.get(name, 0.0)) for name in SIGNAL_NAMES})

    @classmethod
    def from_file(cls, path: str | Path) -> "RewardConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = data.get("weights", data)
        if not isinstance(weights, dict):
            raise ValueError("reward config must contain a weights object")
        return cls(weights=weights)

    def calculate(self, signals: Mapping[str, float]) -> float:
        missing = set(SIGNAL_NAMES) - set(signals)
        if missing:
            raise ValueError(f"missing reward signals: {sorted(missing)}")
        return float(sum(self.weights[name] * float(signals[name]) for name in SIGNAL_NAMES))
