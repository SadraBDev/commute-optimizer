"""
visualization/interactive_traffic.py
Interactive network visualizer with dynamic time slider and click-to-inspect edge profiles.
"""

from typing import Dict, Tuple, List, Optional, Any
import matplotlib as mpl

try:
    mpl.use("TkAgg")
except Exception:
    try:
        mpl.use("QtAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.widgets import Slider
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx

from core.graph import RoadNetwork
from core.congestion import CongestionModel


def format_minutes_to_time(minutes: float) -> str:
    """Converts minutes from midnight to HH:MM format."""
    total_min = int(round(minutes))
    hours = (total_min // 60) % 24
    mins = total_min % 60
    return f"{hours:02d}:{mins:02d}"


def point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Calculates perpendicular Euclidean distance from point (px, py) to segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


class InteractiveTrafficVisualizer:
    def __init__(
        self,
        network: RoadNetwork,
        congestion_model: CongestionModel,
        t_min: float = 420.0,
        t_max: float = 600.0,
        t_step: float = 5.0
    ):
        self.network = network
        self.congestion_model = congestion_model
        self.t_min = t_min
        self.t_max = t_max
        self.t_step = t_step
        self.current_time = t_min

                           
        fig, ax = plt.subplots(figsize=(13, 8))
        self.fig: Figure = fig
        self.ax: Axes = ax
        self.fig.subplots_adjust(bottom=0.18)

        self.g = nx.DiGraph()
        self.pos: Dict[str, Tuple[float, float]] = {}
        self._init_network_layout()

        self.cmap = mpl.colormaps["RdYlGn_r"]
        self.norm = mcolors.Normalize(vmin=0.0, vmax=4.0)

        self.edge_artists: Dict[Tuple[str, str], Any] = {}
        self.edge_text_artists: Dict[Tuple[str, str], Text] = {}
        self.selected_edge_patch: Optional[Any] = None

                                       
        self.profile_fig: Optional[Figure] = None
        self.profile_vlink: Optional[Line2D] = None
        self.active_inspected_edge: Optional[Tuple[str, str]] = None

        self._build_network_artists()
        self._build_slider()
        self._bind_events()

    def _init_network_layout(self) -> None:
        has_coords = False
        for node_id in self.network.all_nodes():
            self.g.add_node(node_id)
            x, y = self.network.get_coordinates(node_id)
            if x != 0.0 or y != 0.0:
                has_coords = True
            self.pos[node_id] = (float(x), float(y))

        if not has_coords:
            raw_pos = nx.spring_layout(self.g, seed=42)
            self.pos = {
                str(node): (float(coords[0]), float(coords[1]))
                for node, coords in raw_pos.items()
            }

        for u, v, _ in self.network.all_edges():
            self.g.add_edge(u, v)

    def _build_network_artists(self) -> None:
        nx.draw_networkx_nodes(
            self.g,
            self.pos,
            node_color="#ECEFF1",
            edgecolors="#263238",
            node_size=1300,
            linewidths=2.0,
            ax=self.ax
        )
        labels = {n: self.network.nodes[n].get("label", n) for n in self.g.nodes()}
        nx.draw_networkx_labels(
            self.g, self.pos, labels=labels, font_size=8, font_weight="bold", ax=self.ax
        )

        edges = [(u, v) for u, v, _ in self.network.all_edges()]
        for u, v in edges:
            patch = nx.draw_networkx_edges(
                self.g,
                self.pos,
                edgelist=[(u, v)],
                edge_color="green",
                arrows=True,
                arrowsize=18,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.08",
                width=3.2,
                ax=self.ax
            )
            self.edge_artists[(u, v)] = patch

            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dx, dy = x2 - x1, y2 - y1
            length = (dx * dx + dy * dy) ** 0.5 + 1e-6
            offset_x, offset_y = -dy / length * 0.28, dx / length * 0.28

            txt = self.ax.text(
                xm + offset_x,
                ym + offset_y,
                "",
                fontsize=7.5,
                fontweight="bold",
                color="#212121",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="#B0BEC5"),
                ha="center",
                va="center"
            )
            self.edge_text_artists[(u, v)] = txt

        self.ax.set_title(
            "Dynamic Network Traffic | Click any edge to inspect congestion profile",
            fontsize=12,
            fontweight="bold"
        )
        self.ax.axis("off")

        sm = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.ax, fraction=0.025, pad=0.03)
        cbar.set_label("Congestion Factor $C_e(t)$", rotation=270, labelpad=14)

    def _build_slider(self) -> None:
        slider_rect: Tuple[float, float, float, float] = (0.20, 0.06, 0.60, 0.04)
        ax_slider = self.fig.add_axes(slider_rect, facecolor="#ECEFF1")
        self.time_slider = Slider(
            ax=ax_slider,
            label="Departure Time: ",
            valmin=self.t_min,
            valmax=self.t_max,
            valinit=self.t_min,
            valstep=self.t_step,
            valfmt="%0.0f min"
        )
        self.time_slider.on_changed(self._on_slider_update)

    def _bind_events(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def _on_slider_update(self, val: float) -> None:
        self.current_time = float(self.time_slider.val)
        time_str = format_minutes_to_time(self.current_time)

        for (u, v), patch in self.edge_artists.items():
            base_time = self.network.get_base_time(u, v)
            mult = self.congestion_model.get_multiplier(u, v, self.current_time)
            dynamic_time = base_time * (1.0 + mult)

            color = self.cmap(self.norm(mult))
            if isinstance(patch, list):
                for p in patch:
                    p.set_edgecolor(color)
            else:
                patch.set_edgecolor(color)

            txt = f"{dynamic_time:.1f}m\n(+{int(mult * 100)}%)" if mult > 0.05 else f"{dynamic_time:.1f}m"
            self.edge_text_artists[(u, v)].set_text(txt)

        self.ax.set_title(
            f"Dynamic Network Traffic | Time: {time_str} ({self.current_time:.1f} min) | (Click any road segment to inspect)",
            fontsize=12,
            fontweight="bold"
        )
        self.fig.canvas.draw_idle()

                                                              
        if self.profile_vlink is not None and self.profile_fig is not None:
            self.profile_vlink.set_xdata([self.current_time, self.current_time])
            self.profile_fig.canvas.draw_idle()

    def _clear_selected_edge(self) -> None:
        """Removes the cyan highlight from the selected edge."""
        if self.selected_edge_patch is not None:
            try:
                if isinstance(self.selected_edge_patch, list):
                    for p in self.selected_edge_patch:
                        p.remove()
                elif hasattr(self.selected_edge_patch, "remove"):
                    self.selected_edge_patch.remove()
            except Exception:
                pass
            self.selected_edge_patch = None
            self.active_inspected_edge = None
            self.fig.canvas.draw_idle()

    def _on_profile_window_close(self, event: Any) -> None:
        """Callback triggered when the user closes the edge profile window."""
        self._clear_selected_edge()
        self.profile_fig = None
        self.profile_vlink = None
        self.active_inspected_edge = None

    def _on_canvas_click(self, event: Any) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        click_x, click_y = float(event.xdata), float(event.ydata)
        best_edge = None
        min_distance = float("inf")

        for u, v, _ in self.network.all_edges():
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            dist = point_to_segment_distance(click_x, click_y, x1, y1, x2, y2)
            if dist < min_distance:
                min_distance = dist
                best_edge = (u, v)

                                                                                               
        if best_edge is not None and min_distance < 0.65:
            self._highlight_selected_edge(best_edge)
            self._open_edge_profile_window(best_edge)
        else:
            self._clear_selected_edge()

    def _highlight_selected_edge(self, edge: Tuple[str, str]) -> None:
        self._clear_selected_edge()
        u, v = edge
        self.selected_edge_patch = nx.draw_networkx_edges(
            self.g,
            self.pos,
            edgelist=[(u, v)],
            edge_color="#00E5FF",
            arrows=True,
            arrowsize=24,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08",
            width=5.5,
            ax=self.ax
        )
        self.fig.canvas.draw_idle()

    def _open_edge_profile_window(self, edge: Tuple[str, str]) -> None:
        u, v = edge
        self.active_inspected_edge = edge
        base_time = self.network.get_base_time(u, v)

        steps = 150
        time_samples = [
            self.t_min + i * (self.t_max - self.t_min) / (steps - 1) for i in range(steps)
        ]
        multipliers = [self.congestion_model.get_multiplier(u, v, t) for t in time_samples]
        travel_times = [base_time * (1.0 + m) for m in multipliers]

        if self.profile_fig is None:
            fig, _ = plt.subplots(figsize=(8, 5))
            self.profile_fig = fig
            self.profile_fig.canvas.mpl_connect("close_event", self._on_profile_window_close)
            mgr = self.profile_fig.canvas.manager
            if mgr is not None and hasattr(mgr, "set_window_title"):
                mgr.set_window_title("Edge Profile Inspector")
        else:
            self.profile_fig.clf()

        ax1: Axes = self.profile_fig.add_subplot(111)
        ax2: Axes = ax1.twinx()

        line1 = ax1.plot(time_samples, travel_times, color="#D32F2F", linewidth=2.5, label="Travel Time (min)")
        line2 = ax2.plot(time_samples, multipliers, color="#FF9800", linestyle="--", linewidth=2.0, label="Multiplier C_e(t)")

        vlink: Line2D = ax1.axvline(
            x=self.current_time, color="#1976D2", linestyle=":", linewidth=2.2, label="Current Slider Time"
        )
        self.profile_vlink = vlink

        ax1.set_xlabel("Time of Day (Minutes from midnight)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Edge Duration \u03c4(e, t) (minutes)", color="#D32F2F", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Congestion Multiplier C_e(t)", color="#FF9800", fontsize=10, fontweight="bold")

        ticks = [self.t_min + i * 30.0 for i in range(int((self.t_max - self.t_min) // 30) + 1)]
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([format_minutes_to_time(t) for t in ticks], rotation=25)

        lines: List[Line2D] = list(line1) + list(line2) + [vlink]
        labels: List[str] = [str(l.get_label()) for l in lines]
        ax1.legend(lines, labels, loc="upper right", framealpha=0.9)

        ax1.set_title(
            f"Congestion Profile: Edge ({u} \u2192 {v}) | Base Time: {base_time:.1f} min",
            fontsize=11,
            fontweight="bold"
        )
        ax1.grid(True, linestyle="--", alpha=0.5)
        self.profile_fig.tight_layout()
        self.profile_fig.canvas.draw_idle()
        plt.show(block=False)

    def show(self) -> None:
        self._on_slider_update(self.t_min)
        plt.show()


def launch_interactive_traffic_viewer(
    network: RoadNetwork,
    congestion_model: CongestionModel,
    t_min: float = 420.0,
    t_max: float = 600.0,
    t_step: float = 5.0
) -> None:
    viewer = InteractiveTrafficVisualizer(network, congestion_model, t_min, t_max, t_step)
    viewer.show()


if __name__ == "__main__":
    import os
    from core.congestion import GaussianPeakCongestion

    map_path = os.path.join("data", "maps", "metropolis_graph.json")
    if not os.path.exists(map_path):
        map_path = os.path.join("data", "maps", "toy_graph.json")

    scenario_path = os.path.join("data", "scenarios", "dual_peak_rush_hour.json")
    if not os.path.exists(scenario_path):
        scenario_path = os.path.join("data", "scenarios", "rush_hour.json")

    net = RoadNetwork.load_json(map_path)
    model = GaussianPeakCongestion.load_json(scenario_path)
    launch_interactive_traffic_viewer(net, model)
