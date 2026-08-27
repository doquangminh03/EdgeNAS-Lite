from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Target:
    task: str
    device: str
    dataset: str


@dataclass(frozen=True)
class Constraints:
    minimum_map50_95: float
    maximum_median_latency_ms: float
    maximum_model_size_mb: float


@dataclass(frozen=True)
class Preferences:
    optimization_goal: str


@dataclass(frozen=True)
class Requirement:
    schema_version: str
    request_id: str
    target: Target
    constraints: Constraints
    preferences: Preferences

    def to_dict(self) -> Dict[str, Any]:
        """Convert the validated requirement into a dictionary."""

        return asdict(self)