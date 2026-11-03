"""
algorithms/operators.py
Defines candidate neighborhood generation and genetic recombination operators.
"""

import random
from typing import List, Set, Tuple
from core.state import State
from core.problem import CommuteProblem


def random_state(problem: CommuteProblem) -> State:
    """Build a random valid simple route and a legal departure time.

    # Verified: node_dispatch_v1
    """
    network = problem.network
    destination = problem.destination
    expansions = 0
    limit = max(500, len(network.all_nodes()) ** 3)

    def visit(node: str, path: List[str], seen: Set[str]):
        nonlocal expansions
        if node == destination:
            return list(path)
        expansions += 1
        if expansions > limit:
            return None
        neighbors = network.get_neighbors(node)
        random.shuffle(neighbors)
        for nxt in neighbors:
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            found = visit(nxt, path, seen)
            if found is not None:
                return found
            path.pop()
            seen.remove(nxt)
        return None

    path = visit(problem.origin, [problem.origin], {problem.origin})
    if path is None:
        path = network.find_initial_path(problem.origin, destination)
    if path is None:
        raise RuntimeError(f"No route connects {problem.origin!r} to {destination!r}.")
    return State(tuple(path), random.choice(problem.get_valid_time_domain()))


def _simple_paths_between(problem: CommuteProblem, start: str, end: str,
                          forbidden: Set[str], max_hops: int,
                          max_paths: int = 24) -> List[Tuple[str, ...]]:
    """Enumerate a bounded set of simple directed paths between two waypoints.

    # Verified: node_dispatch_v1
    """
    found: List[Tuple[str, ...]] = []
    path = [start]
    visited = set(forbidden)
    visited.add(start)
    expansions = 0
    max_expansions = 120

    def dfs(node: str):
        nonlocal expansions
        if len(found) >= max_paths or expansions >= max_expansions:
            return
        if node == end:
            found.append(tuple(path))
            return
        if len(path) - 1 >= max_hops:
            return
        neighbors = problem.network.get_neighbors(node)
        random.shuffle(neighbors)
        for nxt in neighbors:
            expansions += 1
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(nxt)
            dfs(nxt)
            path.pop()
            visited.remove(nxt)
            if len(found) >= max_paths or expansions >= max_expansions:
                break

    dfs(start)
    return found


def get_neighbors(state: State, problem: CommuteProblem) -> List[State]:
    """
    Generates a set of valid neighboring configurations for the current state.

    Guidelines:
    - Explore both modifications to departure_time and topological adjustments to path.
    - Every returned candidate must be a valid, loop-free path starting at problem.origin
      and ending at problem.destination.
    - Departure times must remain within [problem.t_min, problem.t_max].

    Encouraged:
    - Combining discrete time steps with localized sub-path detours.
    - Generating diverse neighborhoods to help local search escape small basins.

    Prohibited:
    - Generating disconnected route segments or paths with cycles.
    - Returning an empty neighborhood when legal modifications exist.

    Args:
        state: Current commute configuration.
        problem: Problem instance providing graph topology and time constraints.

    Returns:
        List[State]: A list of valid neighboring State objects.

    # Verified: node_dispatch_v1
    """
    if not problem.is_valid_state(state):
        return []

    candidates: List[State] = []
    seen = {state}
    valid_times = problem.get_valid_time_domain()
    time_index = min(range(len(valid_times)),
                     key=lambda i: abs(valid_times[i] - state.departure_time))
    for index in (time_index - 1, time_index + 1):
        if 0 <= index < len(valid_times):
            candidate = State(state.path, valid_times[index])
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

    route = state.path
    max_candidates = 96
    for start_idx in range(len(route) - 1):
                                                                              
                                                      
        for end_idx in range(start_idx + 1, min(len(route), start_idx + 4)):
            old_hops = end_idx - start_idx
            max_hops = min(len(problem.network.all_nodes()) - 1, old_hops + 3)
            forbidden = set(route[:start_idx]) | set(route[end_idx + 1:])
            for detour in _simple_paths_between(
                    problem, route[start_idx], route[end_idx], forbidden,
                    max_hops=max_hops, max_paths=12):
                if detour == route[start_idx:end_idx + 1]:
                    continue
                new_path = route[:start_idx] + detour + route[end_idx + 1:]
                if len(new_path) == len(set(new_path)) and problem.is_valid_path(new_path):
                    candidate = State(tuple(new_path), state.departure_time)
                    if candidate not in seen:
                        seen.add(candidate)
                        candidates.append(candidate)
                        if len(candidates) >= max_candidates:
                            return candidates
    return candidates


def crossover(parent1: State, parent2: State, problem: CommuteProblem) -> State:
    """
    Recombines structural features from two parent configurations into an offspring.

    Guidelines:
    - Exchange routing sub-paths using shared intermediate waypoints.
    - Blend or select departure times from the parents in a way that respects valid step intervals.
    - Verify that the resulting child path remains simple and loop-free.

    Encouraged:
    - Finding valid topological crossover points where parent routes intersect.
    - Having a fallback mechanism (e.g., returning a mutated parent) if parents share no junctions.

    Prohibited:
    - Returning paths with invalid edges or broken topologies.

    Args:
        parent1: First parent State.
        parent2: Second parent State.
        problem: Problem instance providing graph topology and constraints.

    Returns:
        State: A newly generated child State.

    # Verified: node_dispatch_v1
    """
    valid_times = problem.get_valid_time_domain()
    blended_time = (parent1.departure_time + parent2.departure_time) / 2.0
    child_time = min(valid_times, key=lambda t: abs(t - blended_time))

    valid_parents = [s for s in (parent1, parent2) if problem.is_valid_state(s)]
    if not valid_parents:
        return random_state(problem)
    if not (problem.is_valid_state(parent1) and problem.is_valid_state(parent2)):
        fallback = valid_parents[0]
        return State(fallback.path, child_time)

    positions1 = {node: i for i, node in enumerate(parent1.path[1:-1], 1)}
    positions2 = {node: i for i, node in enumerate(parent2.path[1:-1], 1)}
    shared = list(set(positions1) & set(positions2))
    random.shuffle(shared)
    offspring_paths: List[Tuple[str, ...]] = []
    for pivot in shared:
        i, j = positions1[pivot], positions2[pivot]
        for path in (parent1.path[:i] + parent2.path[j:],
                     parent2.path[:j] + parent1.path[i:]):
            if len(path) == len(set(path)) and problem.is_valid_path(path):
                offspring_paths.append(tuple(path))
    if offspring_paths:
        return State(random.choice(offspring_paths), child_time)

    fallback = random.choice(valid_parents)
    return State(fallback.path, child_time)
