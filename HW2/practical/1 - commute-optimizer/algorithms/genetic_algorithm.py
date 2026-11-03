"""
algorithms/genetic_algorithm.py
Population-based optimization using selection, recombination, and mutation.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover, random_state
import random


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        Guidelines:
        - Initialize a population of valid candidate solutions.
        - Design a fitness metric suitable for minimizing commute travel time (lower cost -> higher fitness).
        - Use selection operators (such as tournament or roulette selection) to pick parents.
        - Pair parents to produce offspring via `crossover()`, and apply stochastic mutation.
        - Preserve the top candidates across generations if using elitism.
        - Track best population performance per generation in self.history.

        Encouraged:
        - Experimenting with population size, selection pressure, and operator rates to balance exploration vs. exploitation.
        """
        size = max(2, int(self.population_size))
        generations = max(0, int(self.generations))
        crossover_rate = min(1.0, max(0.0, float(self.crossover_rate)))
        mutation_rate = min(1.0, max(0.0, float(self.mutation_rate)))
        elite_count = min(size, max(0, int(self.elitism)))

        population = [self.problem.generate_initial_state()]
        attempts = 0
        while len(population) < size:
            attempts += 1
            candidate = random_state(self.problem)
            if candidate not in population or attempts >= size * 8:
                population.append(candidate)

        costs = {state: self.problem.evaluate(state) for state in population}
        best_state = min(population, key=lambda state: costs[state])
        best_cost = costs[best_state]
        self.history = [best_cost]

        def tournament() -> State:
            contestants = random.sample(population, k=min(3, len(population)))
            return min(contestants, key=lambda state: costs[state])

        for _ in range(generations):
            ordered = sorted(population, key=lambda state: costs[state])
            next_population = ordered[:elite_count]
            while len(next_population) < size:
                parent1, parent2 = tournament(), tournament()
                child = (crossover(parent1, parent2, self.problem)
                         if random.random() < crossover_rate else random.choice((parent1, parent2)))
                if random.random() < mutation_rate:
                    neighbors = get_neighbors(child, self.problem)
                    if neighbors:
                        child = random.choice(neighbors)
                if not self.problem.is_valid_state(child):
                    child = random.choice((parent1, parent2))
                next_population.append(child)

            population = next_population
            costs = {state: self.problem.evaluate(state) for state in population}
            generation_best = min(population, key=lambda state: costs[state])
            if costs[generation_best] < best_cost:
                best_state, best_cost = generation_best, costs[generation_best]
            self.history.append(best_cost)

        return best_state, best_cost
