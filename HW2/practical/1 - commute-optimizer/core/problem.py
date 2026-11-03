"""
core/problem.py
Evaluates candidate commute states and simulates dynamic travel times.
"""

from typing import Optional, List, Tuple
from core.state import State
from core.graph import RoadNetwork
from core.congestion import CongestionModel


class CommuteProblem:
    def __init__(
        self,
        network: RoadNetwork,
        congestion_model: CongestionModel,
        origin: str,
        destination: str,
        t_min: float = 420.0,                        
        t_max: float = 600.0,                        
        t_step: float = 5.0                          
    ):
        if not network.has_node(origin):
            raise KeyError(f"Origin node '{origin}' not found in network.")
        if not network.has_node(destination):
            raise KeyError(f"Destination node '{destination}' not found in network.")
        if t_min >= t_max:
            raise ValueError(f"t_min ({t_min}) must be strictly less than t_max ({t_max}).")
        if t_step <= 0:
            raise ValueError(f"t_step ({t_step}) must be positive.")

        self.network = network
        self.congestion_model = congestion_model
        self.origin = origin
        self.destination = destination
        self.t_min = t_min
        self.t_max = t_max
        self.t_step = t_step

    def get_valid_time_domain(self) -> List[float]:
        """Returns the list of valid discrete departure timestamps."""
        times = []
        curr = self.t_min
        while curr <= self.t_max + 1e-9:
            times.append(round(curr, 2))
            curr += self.t_step
        return times

    def is_valid_path(self, path: tuple) -> bool:
        """
        Validates that a path is non-empty, loop-free, starts at origin,
        ends at destination, and consists solely of existing network edges.
        """
        if not path or path[0] != self.origin or path[-1] != self.destination:
            return False
        if len(path) != len(set(path)):
            return False                   
        for i in range(len(path) - 1):
            if not self.network.has_edge(path[i], path[i + 1]):
                return False
        return True

    def is_valid_state(self, state: State) -> bool:
        """Validates both path topology and departure time window bounds."""
        if not (self.t_min <= state.departure_time <= self.t_max):
            return False
        return self.is_valid_path(state.path)

    def evaluate_edge_travel_time(self, u: str, v: str, current_time: float) -> float:
        """Calculates dynamic travel time along a single edge at a specific timestamp."""
        base_time = self.network.get_base_time(u, v)
        congestion = self.congestion_model.get_multiplier(u, v, current_time)
        return base_time * (1.0 + congestion)

    def evaluate(self, state: State) -> float:
        """
        Objective function f(x): Computes total commute travel time (minutes).
        Returns infinity (float('inf')) if state is invalid.
        """
        if not self.is_valid_state(state):
            return float('inf')

        current_time = state.departure_time
        for i in range(len(state.path) - 1):
            u, v = state.path[i], state.path[i + 1]
            edge_duration = self.evaluate_edge_travel_time(u, v, current_time)
            current_time += edge_duration

        return current_time - state.departure_time

    def get_path_cost_vector(self, state: State) -> Tuple[float, int]:
        """Returns (commute_time, hops_count). Use index [0] for objective cost."""
        ...
    
    def generate_initial_state(self) -> State:
        """
        Constructs a valid baseline state using the shortest free-flow path
        and the earliest departure timestamp (t_min).
        """
        path = self.network.find_initial_path(self.origin, self.destination)
        if path is None:
            raise RuntimeError(f"No path connects '{self.origin}' to '{self.destination}'.")
        return State(path=tuple(path), departure_time=self.t_min)
