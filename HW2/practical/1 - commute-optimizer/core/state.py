"""
core/state.py
Immutable configuration state x = (Path, DepartureTime) for local search.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class State:
    """
    Represents an immutable complete state x in the local search space.
    - path: Ordered tuple of node identifiers forming a path from origin to destination.
    - departure_time: Timestamp in minutes from midnight (e.g., 480.0 = 08:00 AM).
    """
    path: Tuple[str, ...]
    departure_time: float

    def __post_init__(self):
        if not isinstance(self.path, tuple):
            object.__setattr__(self, "path", tuple(self.path))
        object.__setattr__(self, "departure_time", float(self.departure_time))

    def format_time(self) -> str:
        """Returns HH:MM string representation of the departure time."""
        total_minutes = int(round(self.departure_time))
        hours = (total_minutes // 60) % 24
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"

    def __repr__(self) -> str:
        route_display = " -> ".join(self.path)
        return f"State(Depart: {self.format_time()} [{self.departure_time:.1f}m], Path: [{route_display}])"
