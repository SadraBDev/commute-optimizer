"""
algorithms/base.py
Abstract base class for local search solvers.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Dict
from core.state import State
from core.problem import CommuteProblem


class LocalSearchSolver(ABC):
    def __init__(self, problem: CommuteProblem, **kwargs):
        self.problem = problem
        self.params: Dict[str, Any] = kwargs
        self.history: List[float] = []

    @abstractmethod
    def solve(self) -> Tuple[State, float]:
        """
        Executes the search procedure.

        Returns:
            Tuple[State, float]: The best State found and its evaluated objective cost.
        """
        pass
