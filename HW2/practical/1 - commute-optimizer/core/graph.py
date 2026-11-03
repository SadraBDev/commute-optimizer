"""
core/graph.py
Represents a directed road network with spatial coordinates and free-flow travel times.
"""

from typing import Dict, List, Tuple, Optional, Any
import json
import heapq


class RoadNetwork:
    def __init__(self):
                                                           
        self.nodes: Dict[str, Dict[str, Any]] = {}
                                    
        self.adj: Dict[str, Dict[str, float]] = {}

    def add_node(self, node_id: str, x: float = 0.0, y: float = 0.0, label: Optional[str] = None) -> None:
        """Adds an intersection/station node with optional 2D layout coordinates."""
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "x": float(x),
                "y": float(y),
                "label": label if label is not None else node_id
            }
            self.adj[node_id] = {}

    def add_edge(self, u: str, v: str, base_time: float) -> None:
        """
        Adds a directed road segment from node u to node v with free-flow travel time.
        Raises ValueError if base_time <= 0 or if nodes do not exist.
        """
        if u not in self.nodes or v not in self.nodes:
            raise KeyError(f"Both nodes must exist before adding edge: ({u} -> {v})")
        if base_time <= 0:
            raise ValueError(f"Base travel time must be positive. Got {base_time} for ({u} -> {v})")
        self.adj[u][v] = float(base_time)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def has_edge(self, u: str, v: str) -> bool:
        return u in self.adj and v in self.adj[u]

    def get_base_time(self, u: str, v: str) -> float:
        """Returns baseline free-flow travel time between u and v."""
        if not self.has_edge(u, v):
            raise KeyError(f"Edge ({u} -> {v}) does not exist in network.")
        return self.adj[u][v]

    def get_neighbors(self, u: str) -> List[str]:
        """Returns all outbound adjacent nodes from u."""
        return list(self.adj.get(u, {}).keys())

    def get_coordinates(self, node_id: str) -> Tuple[float, float]:
        """Returns (x, y) coordinates for spatial plotting."""
        node = self.nodes.get(node_id)
        if not node:
            raise KeyError(f"Node '{node_id}' does not exist.")
        return node["x"], node["y"]

    def all_nodes(self) -> List[str]:
        return list(self.nodes.keys())

    def all_edges(self) -> List[Tuple[str, str, float]]:
        """Returns a list of tuples: (u, v, base_time)."""
        edges = []
        for u, neighbors in self.adj.items():
            for v, base_time in neighbors.items():
                edges.append((u, v, base_time))
        return edges

    def find_initial_path(self, origin: str, destination: str) -> Optional[List[str]]:
        """
        Finds a baseline valid, loop-free path from origin to destination 
        using Dijkstra's algorithm on free-flow edge times.
        Returns None if no connected path exists.
        """
        if not self.has_node(origin) or not self.has_node(destination):
            return None

        distances = {node: float('inf') for node in self.nodes}
        distances[origin] = 0.0
        previous: Dict[str, Optional[str]] = {node: None for node in self.nodes}
        pq: List[Tuple[float, str]] = [(0.0, origin)]

        while pq:
            curr_dist, u = heapq.heappop(pq)
            if curr_dist > distances[u]:
                continue
            if u == destination:
                break

            for v, base_time in self.adj[u].items():
                new_dist = curr_dist + base_time
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    previous[v] = u
                    heapq.heappush(pq, (new_dist, v))

        if distances[destination] == float('inf'):
            return None

                          
        path = []
        curr: Optional[str] = destination
        while curr is not None:
            path.append(curr)
            curr = previous[curr]
        path.reverse()
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Serializes network to a native dictionary."""
        return {
            "nodes": [
                {"id": nid, "x": data["x"], "y": data["y"], "label": data["label"]}
                for nid, data in self.nodes.items()
            ],
            "edges": [
                {"u": u, "v": v, "base_time": weight}
                for u, neighbors in self.adj.items()
                for v, weight in neighbors.items()
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoadNetwork":
        """Instantiates a RoadNetwork from a dictionary."""
        net = cls()
        for n in data.get("nodes", []):
            net.add_node(n["id"], n.get("x", 0.0), n.get("y", 0.0), n.get("label"))
        for e in data.get("edges", []):
            net.add_edge(e["u"], e["v"], e["base_time"])
        return net

    def save_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "RoadNetwork":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
