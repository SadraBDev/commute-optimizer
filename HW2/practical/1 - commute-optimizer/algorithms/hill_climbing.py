"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
import random
from algorithms.operators import get_neighbors, random_state


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Hill Climbing optimization.

        Guidelines:
        - Search through the neighborhood greedily for candidates that minimize total travel time.
        - Handle flat local maxima / shoulders by allowing up to `max_sideways` non-improving steps.
        - Apply up to `max_restarts` to explore different starting basins.
        - Track the best objective cost seen at each major step in self.history.

        Prohibited:
        - Looping indefinitely on flat plateaus.
        """
        max_steps = max(1, int(self.params.get("max_iterations", 500)))
        best_state = self.problem.generate_initial_state()
        best_cost = self.problem.evaluate(best_state)
        self.history = [best_cost]

        current_start = best_state
        for restart in range(max(0, self.max_restarts) + 1):
            current = current_start
            current_cost = self.problem.evaluate(current)
            sideways_used = 0
            visited = {current}

            for _ in range(max_steps):
                neighbors = [s for s in get_neighbors(current, self.problem)
                             if s not in visited]
                if not neighbors:
                    break
                scored = [(self.problem.evaluate(candidate), candidate)
                          for candidate in neighbors]
                next_cost = min(cost for cost, _ in scored)
                tied = [state for cost, state in scored
                        if abs(cost - next_cost) <= 1e-9]
                next_state = random.choice(tied)

                if next_cost < current_cost - 1e-9:
                    current, current_cost = next_state, next_cost
                    sideways_used = 0
                elif (abs(next_cost - current_cost) <= 1e-9
                      and sideways_used < max(0, self.max_sideways)):
                    current, current_cost = next_state, next_cost
                    sideways_used += 1
                else:
                    break

                visited.add(current)
                if current_cost < best_cost:
                    best_state, best_cost = current, current_cost
                self.history.append(best_cost)

            if restart < max(0, self.max_restarts):
                                                                            
                                                              
                current_start = random_state(self.problem)
                restart_cost = self.problem.evaluate(current_start)
                if restart_cost < best_cost:
                    best_state, best_cost = current_start, restart_cost
                self.history.append(best_cost)
        return best_state, best_cost
