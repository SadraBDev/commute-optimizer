"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        Guidelines:
        - Maintain a set of `beam_width` parallel configurations.
        - In each iteration, generate neighbors from all active states in the beam.
        - Pool and rank all generated successors collectively, then retain the top `beam_width` states.
        - Halt when the beam ceases to improve or reaches max_iterations.
        - Record the best cost of the population at each step in self.history.

        Encouraged:
        - Ensuring diversity across states in the beam to prevent collapse into a single redundant path.

        Prohibited:
        - Treating this as K independent, non-communicating searches.
        """
        width = max(1, int(self.beam_width))
        initial = self.problem.generate_initial_state()
        beam = [initial]
        attempts = 0
        while len(beam) < width and attempts < width * 10:
            attempts += 1
            candidate = random_state(self.problem)
            if candidate not in beam:
                beam.append(candidate)

        scored = sorted(((self.problem.evaluate(state), state) for state in beam),
                        key=lambda item: item[0])
        best_cost, best_state = scored[0]
        self.history = [best_cost]
        previous_signature = None

        for _ in range(max(0, int(self.max_iterations))):
            pool = {}
            for state in beam:
                for candidate in get_neighbors(state, self.problem):
                    pool[candidate] = self.problem.evaluate(candidate)
            if not pool:
                break

            ranked = sorted(((cost, state) for state, cost in pool.items()),
                            key=lambda item: item[0])
            next_beam = []
            route_counts = {}
                                                                             
                                                                          
            for _, candidate in ranked:
                count = route_counts.get(candidate.path, 0)
                if count == 0:
                    next_beam.append(candidate)
                    route_counts[candidate.path] = 1
                if len(next_beam) == width:
                    break
            if len(next_beam) < width:
                for _, candidate in ranked:
                    if candidate not in next_beam:
                        next_beam.append(candidate)
                        if len(next_beam) == width:
                            break
            next_beam = next_beam or beam

            generation_best = min(self.problem.evaluate(state) for state in next_beam)
            if generation_best < best_cost:
                best_cost = generation_best
                best_state = min(next_beam, key=self.problem.evaluate)
            self.history.append(best_cost)
            signature = tuple(sorted((state.path, state.departure_time)
                                     for state in next_beam))
            if signature == previous_signature or set(next_beam) == set(beam):
                break
            previous_signature = signature
            beam = next_beam

        return best_state, best_cost
