"""
core/congestion.py
Dynamic, time-dependent traffic congestion models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import math
import json


class CongestionModel(ABC):
    @abstractmethod
    def get_multiplier(self, u: str, v: str, current_time: float) -> float:
        """
        Returns dynamic traffic congestion factor C_e(t) >= 0 for edge (u -> v).
        Total edge traversal time will be: base_time * (1.0 + C_e(t)).
        """
        pass


class GaussianPeakCongestion(CongestionModel):
    """
    Models rush-hour waves using Gaussian distributions on specific edges.
    peaks: dict mapping (u, v) -> list of peak definitions:
           [{"peak_time": float, "sigma": float, "amplitude": float}, ...]
    """
    def __init__(self, peaks: Dict[Tuple[str, str], list]):
        self.peaks = peaks

    def get_multiplier(self, u: str, v: str, current_time: float) -> float:
        edge = (u, v)
        if edge not in self.peaks:
            return 0.0

        multiplier = 0.0
        for peak in self.peaks[edge]:
            mu = peak["peak_time"]
            sigma = peak["sigma"]
            amp = peak["amplitude"]
            if sigma > 0:
                multiplier += amp * math.exp(-((current_time - mu) ** 2) / (2.0 * (sigma ** 2)))
        return max(0.0, multiplier)

    @classmethod
    def load_json(cls, filepath: str) -> "GaussianPeakCongestion":
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        peaks = {}
        for item in raw_data.get("peaks", []):
            edge = (item["u"], item["v"])
            peak_params = {
                "peak_time": float(item["peak_time"]),
                "sigma": float(item["sigma"]),
                "amplitude": float(item["amplitude"])
            }
            if edge not in peaks:
                peaks[edge] = []
            peaks[edge].append(peak_params)

        return cls(peaks)
