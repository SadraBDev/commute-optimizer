"""
visualization/plot_routes.py
Visualizes the road network, traffic congestion heat, and selected commute route.
"""

from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx

from core.state import State
from core.problem import CommuteProblem


def plot_commute_route(
    problem: CommuteProblem,
    state: State,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Renders the road network with spatial coordinates:
    - Base road colors reflect traffic congestion intensity at departure_time.
    - The chosen route is highlighted in bold crimson.
    """
    network = problem.network
    g = nx.DiGraph()

                                         
    pos = {}
    has_custom_coords = False
    for node_id in network.all_nodes():
        g.add_node(node_id)
        x, y = network.get_coordinates(node_id)
        if x != 0.0 or y != 0.0:
            has_custom_coords = True
        pos[node_id] = (x, y)

    if not has_custom_coords:
        pos = nx.spring_layout(g, seed=42)

                                                              
    edges = []
    congestion_multipliers = []
    for u, v, _ in network.all_edges():
        g.add_edge(u, v)
        edges.append((u, v))
        mult = problem.congestion_model.get_multiplier(u, v, state.departure_time)
        congestion_multipliers.append(mult)

                                                         
    max_c = max(congestion_multipliers) if congestion_multipliers and max(congestion_multipliers) > 0 else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, max_c))
    cmap = cm.YlOrRd

    plt.figure(figsize=(11, 7))
    ax = plt.gca()

                                                      
    edge_colors = [cmap(norm(c)) for c in congestion_multipliers]
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=edges,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=18,
        arrowstyle="-|>",
        node_size=1200,
        connectionstyle="arc3,rad=0.08",
        width=2.5,
        ax=ax
    )

                                               
    route_edges = []
    for i in range(len(state.path) - 1):
        route_edges.append((state.path[i], state.path[i + 1]))

    if route_edges:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=route_edges,
            edge_color="#0066FF",
            arrows=True,
            arrowsize=22,
            arrowstyle="-|>",
            node_size=1200,
            connectionstyle="arc3,rad=0.08",
            width=5.0,
            ax=ax,
            label="Optimized Route"
        )

                
    node_fill_colors = []
    for n in g.nodes():
        if n == problem.origin:
            node_fill_colors.append("#2ECC71")                        
        elif n == problem.destination:
            node_fill_colors.append("#E74C3C")                      
        elif n in state.path:
            node_fill_colors.append("#85C1E9")                                   
        else:
            node_fill_colors.append("#BDC3C7")                         

    nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_fill_colors,
        node_size=1200,
        edgecolors="#2C3E50",
        linewidths=1.8,
        ax=ax
    )

                 
    labels = {n: network.nodes[n].get("label", n) for n in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=9, font_weight="bold", ax=ax)

                                      
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Dynamic Traffic Factor $C_e(t)$", rotation=270, labelpad=15)

                          
    travel_time = problem.evaluate(state)
    header = title if title else "Optimized Commute Trajectory"
    plt.title(
        f"{header}\nDeparture: {state.format_time()} | Total Time: {travel_time:.2f} min",
        fontsize=12,
        fontweight="bold"
    )
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
