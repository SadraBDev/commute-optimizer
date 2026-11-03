# Commute Optimizer: Urban Route & Departure Scheduling via Local Search

An extensible benchmarking environment and practical assignment for solving dynamic urban commute problems using Classical Artificial Intelligence and Local Search heuristics.

---

## 1. Problem Overview

In classical pathfinding (e.g., Dijkstra, A*), an agent incrementally explores edges from origin to destination. However, in real-world urban networks, travel times are **time-dependent**: departing at 07:45 AM yields drastically different delays compared to departing at 08:30 AM due to dynamic congestion.

This project formulates the commute task as a **Complete-State Configuration Optimization Problem**. Rather than deciding turns step-by-step, the search agent optimizes an entire commute configuration simultaneously:

$$x = \langle P, t_{\text{start}} \rangle$$

where:
* $P = \langle v_0, v_1, \dots, v_m \rangle$ is a complete, loop-free topological path from Origin ($v_0$) to Destination ($v_m$) in the road network.
* $t_{\text{start}}$ is the departure time chosen from a discrete set of options within a commute window: $\mathcal{T} = \{t_{\min} + k \cdot \Delta t \mid k \in \mathbb{N}_0, \, t_{\min} + k \cdot \Delta t \le t_{\max}\}$.

The objective is to discover the configuration $x^*$ that minimizes the total commute duration $f(x)$ under dynamic rush-hour traffic waves.

---

## 2. Mathematical Formulation

### 2.1 The Road Network
The urban layout is represented as a directed graph $G = (V, E)$. Every directed edge $e = (u, v) \in E$ has:
* A baseline free-flow travel time $t_e^{\text{base}} > 0$ (the traversal time under zero traffic).
* Spatial 2D coordinates $(x_v, y_v)$ for visualization.

### 2.2 Dynamic Congestion Model
Each road segment experiences dynamic, time-dependent traffic intensity represented by a non-negative multiplier $C_e(t) \ge 0$. The dynamic travel time $\tau(e, t)$ to traverse edge $e$ when entered at timestamp $t$ is:

$$\tau(e, t) = t_e^{\text{base}} \times \big(1 + C_e(t)\big)$$

Congestion waves are generated using a **Gaussian Rush-Hour Peak Model**:

$$C_e(t) = \sum_{k} A_{e, k} \cdot \exp\left(-\frac{(t - \mu_{e, k})^2}{2\sigma_{e, k}^2}\right)$$

where $\mu$ denotes peak bottleneck time, $\sigma$ controls duration, and $A$ indicates congestion severity.

### 2.3 The Objective Function $f(x)$
Given a candidate configuration $x = \big(\langle v_0, v_1, \dots, v_m \rangle, \, t_{\text{start}}\big)$, the simulation advances the clock along the path using a First-In-First-Out (FIFO) dynamic network simulation:

1. Initialize clock: $t_0 = t_{\text{start}}$
2. For each successive edge $e_i = (v_i, v_{i+1})$ along path $P$, compute dynamic travel time $\Delta t_i = \tau(e_i, t_i)$ and advance the clock:
   $$t_{i+1} = t_i + \Delta t_i$$
3. Total travel duration to minimize:
   $$f(x) = t_m - t_{\text{start}} = \sum_{i=0}^{m-1} \Delta t_i$$

If a state contains cycles, lacks path continuity, or sets a departure time outside $[t_{\min}, t_{\max}]$, the evaluation engine assigns an infinite penalty: $f(x) = \infty$.

---

## 3. Local Search Algorithms

You will implement four local search algorithms:

1. **Hill Climbing (Steepest Descent)**:
   * Greedily transitions to the neighboring state with the lowest travel duration.
   * Handles flat local minima and shoulders by allowing a limited budget of sideways moves (`max_sideways`).
   * Escapes sub-optimal local basins using random restarts (`max_restarts`).

2. **Simulated Annealing**:
   * A stochastic search method governed by an exponential cooling schedule $T_{k+1} = \alpha \cdot T_k$.
   * Improvements ($\Delta E < 0$) are always accepted.
   * Sub-optimal moves ($\Delta E \ge 0$) are accepted probabilistically via the Metropolis criterion:
     $$P(\text{accept}) = \exp\left(-\frac{\Delta E}{T}\right)$$

3. **Local Beam Search**:
   * Simultaneously tracks $K$ candidate commute plans.
   * In each iteration, successors are generated from all $K$ active states and pooled together.
   * The top $K$ configurations from the entire pool are selected for the next iteration.

4. **Genetic Algorithm (GA)**:
   * Maintains a population of valid commute configurations across successive generations.
   * Uses fitness assignment to bias selection (e.g., Tournament or Roulette Wheel) toward faster routes.
   * Performs topological crossover by identifying shared intermediate nodes between parent routes to splice sub-paths and blend departure times.
   * Applies stochastic mutation (detours or time adjustments) to maintain population diversity.

---

## 4. Repository Architecture

```text
commute-optimizer/
│
├── core/                         # [DO NOT MODIFY] Core simulation engine
│   ├── graph.py                  # Directed road network representation & JSON I/O
│   ├── state.py                  # Immutable State(path, departure_time)
│   ├── congestion.py             # Dynamic congestion models
│   └── problem.py                # Dynamic FIFO path evaluator f(x)
│
├── algorithms/                   # [YOUR IMPLEMENTATION AREA]
│   ├── operators.py              # Candidate neighborhood generation & crossover
│   ├── base.py                   # Abstract base solver class
│   ├── hill_climbing.py          # Hill Climbing solver
│   ├── simulated_annealing.py    # Simulated Annealing solver
│   ├── beam_search.py            # Local Beam Search solver
│   └── genetic_algorithm.py      # Genetic Algorithm solver
│
├── visualization/                # Plotting utilities
│   ├── plot_routes.py            # Network renderer with traffic heatmaps
│   ├── plot_metrics.py           # Objective convergence curve plotter
│   └── interactive_traffic.py    # Dynamic slider & edge inspection tool
│
├── data/                         # Maps and scenario files
│   ├── maps/                     # Network topology definitions
│   └── scenarios/                # Congestion profiles and rush-hour bottlenecks
│
├── config.json                   # Centralized hyperparameter configuration
├── main.py                       # CLI execution and benchmarking harness
└── requirements.txt              # Dependencies
```

---

## 5. Benchmark Datasets: Maps & Scenarios

The test suite includes three distinct network maps and five dynamic traffic scenarios designed to evaluate specific theoretical properties of local search heuristics.

### 5.1 Scenario & Map Compatibility Matrix

| Scenario File | Compatible Map | Primary Theoretical Focus | Pedagogical Objective |
| :--- | :--- | :--- | :--- |
| `rush_hour.json` | `toy_graph.json` | **Energy Barrier & Basin Escape** | Tests if Simulated Annealing and Restarts escape the direct highway trap and discover the longer boulevard detour during peak hours. |
| `dual_peak_rush_hour.json` | `metropolis_graph.json` | **Multimodal Surface Exploration** | Multiple overlapping traffic waves create competing local minima, testing stochastic exploration over greedy descents. |
| `cascading_gridlock.json` | `metropolis_graph.json` | **FIFO Downstream Deception** | Penalizes myopic choices on early edges that lead into downstream gridlock, forcing the discovery of mid-route bypass bridges. |
| `crossover_showcase.json` | `metropolis_graph.json` | **Building-Block Recombination** | Evaluates how GA crossover splices partial solutions from two sub-optimal parents to outperform greedy beam search. |
| `flat_plateau.json` | `downtown_grid.json` | **Plateaus & Sideways Moves** | Tests if Hill Climbing leverages `max_sideways` to traverse cost-neutral Manhattan paths without premature termination. |

---

### 5.2 Network Descriptions (`data/maps/`)

1. **`toy_graph.json` (7 Nodes, 12 Edges):**
   * A clean two-corridor setup comparing a short 30-minute Highway against a longer 46-minute Boulevard.
   * Ideal for verifying path validity, debugging operator stubs, and testing basic local basin escapes.
2. **`metropolis_graph.json` (16 Nodes, 26 Edges):**
   * A metropolitan network featuring three parallel corridors (North Expressway, Central Arterial, and South Ring) linked by cross-connecting bridges (`B1`, `B2`) and tunnels (`T1`, `T2`).
   * Provides numerous topological sub-paths, detour alternatives, and common intersection junctions for complex multi-agent benchmarking.
3. **`downtown_grid.json` (12 Nodes, 17 Edges):**
   * A dense $3 \times 4$ urban grid where multiple distinct paths from `Home` to `Uni` share identical baseline hop counts and free-flow travel durations.
   * Specifically designed to generate broad cost-neutral plateaus (shoulders) in the search landscape.

---

### 5.3 Traffic Scenario Details (`data/scenarios/`)

* **`rush_hour.json` (`toy_graph.json`):**
  Applies severe congestion ($A = 4.0$) to Highway links between 07:45 and 08:30 AM ($t \approx 480 \text{--} 510$). While the Highway is optimal at off-peak hours (30 min), during rush hour it escalates to over 85 minutes, making the 46-minute Boulevard the superior choice.
* **`dual_peak_rush_hour.json` (`metropolis_graph.json`):**
  Simulates an early morning school run ($\mu = 450\text{m}$) followed by an office rush ($\mu = 495\text{m}$) on the North Expressway, accompanied by secondary arterial spillovers. Solvers must navigate overlapping waves and dynamic window shifts.
* **`cascading_gridlock.json` (`metropolis_graph.json`):**
  A deceptive scenario where `Home -> N1` appears virtually unconstrained at 07:35 AM ($C_e \approx 0.4$), but severe downstream bottlenecks on `N2 -> N3` and `N3 -> Uni` peak right when the commuter reaches them ($t \approx 495 \text{--} 505$). Solvers must identify cross-bridges (`B1 -> C2 -> T2`) to detour around the downstream shockwave.
* **`crossover_showcase.json` (`metropolis_graph.json`):**
  The North corridor is heavily congested on its exit half (`N2 -> N3 -> Uni`), while the South corridor is blocked on its entrance half (`Home -> S1 -> T1`). Central junction `C2` remains entirely clear. An optimal path cannot be found by mutating either corridor alone; it requires crossing over a North-starting parent with a South-finishing parent at pivot `C2`.
* **`flat_plateau.json` (`downtown_grid.json`):**
  Injects broad, uniform congestion across interior grid intersections, creating large regions where neighboring Manhattan detours yield identical objective costs ($\Delta E = 0$). Tests whether Hill Climbing with sideways moves can navigate across the shoulder to discover the clear perimeter artery.

---

## 6. Getting Started

### Prerequisites
* Python 3.9+
* Recommended: Virtual environment (`venv` or `conda`)

### Fork This Repository
First, <ins>**fork this repository**</ins> to your personal GitHub account so that you can develop your solution within this standard framework.

### Setup
```bash
git clone <your_forked_repository_url>
cd commute-optimizer
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Exploring Dynamic Traffic with the Interactive Viewer
Before writing search code, explore dynamic traffic patterns and bottleneck formations on the road network using the interactive viewer:

```bash
# Launch interactive network viewer on the default toy map
python main.py --interactive

# Launch on the metropolis map with dual-peak traffic
python main.py --interactive --map data/maps/metropolis_graph.json --scenario data/scenarios/dual_peak_rush_hour.json

# Inspect the downtown grid plateau
python main.py --interactive --map data/maps/downtown_grid.json --scenario data/scenarios/flat_plateau.json
```

* **Dynamic Slider:** Drag the slider to observe how edge travel times and bottleneck colors shift across the commute window.
* **Inspect Edge Profiles:** Click directly on any road segment to open its 2D congestion-time curve.
* **Clear Selection:** Click on empty space in the map to dismiss the active edge highlight.

### Running Solvers & Benchmarks
Execute individual algorithms or benchmark all four solvers side-by-side using `main.py`:

```bash
# Run Simulated Annealing on the default map
python main.py --algo sa

# Benchmark all implemented algorithms and display convergence curves
python main.py --algo all --seed 42 --plot-metrics

# Visualize the optimal route chosen by the Genetic Algorithm on the network map
python main.py --algo ga --plot-route

# Run the Genetic Algorithm vs. Beam Search benchmark on the Crossover Showcase
python main.py --algo all --map data/maps/metropolis_graph.json --scenario data/scenarios/crossover_showcase.json --plot-metrics

# Test Hill Climbing sideways moves on the Downtown Flat Plateau
python main.py --algo hc --map data/maps/downtown_grid.json --scenario data/scenarios/flat_plateau.json --plot-metrics

# Run benchmarks on a custom map and save generated plots to disk
python main.py --algo all --map data/maps/toy_graph.json --scenario data/scenarios/rush_hour.json --save-figs

# Test on the challenging cascading gridlock scenario
python main.py --algo all --map data/maps/metropolis_graph.json --scenario data/scenarios/cascading_gridlock.json --plot-metrics
```

#### CLI Arguments
* `--interactive`: Launches the interactive GUI with the time slider to inspect network traffic.
* `--algo`: Solver to run (`hc`, `sa`, `beam`, `ga`, or `all`).
* `--map`: Path to the road network JSON file.
* `--scenario`: Path to the congestion scenario JSON file.
* `--config`: Path to hyperparameter configuration file (defaults to `config.json`).
* `--seed`: Integer seed for deterministic runs.
* `--plot-route`: Renders the road network with traffic heat and the chosen path.
* `--plot-metrics`: Displays convergence curves ($f(x)$ vs. iteration step).
* `--save-figs`: Saves generated plots to disk as PNG images instead of displaying a GUI window.

---

## 7. Hyperparameter Configuration (`config.json`)

To decouple search logic from hardcoded constants, all tunable parameters are stored in `config.json`:

```json
{
  "problem": {
    "origin": "Home",
    "destination": "Uni",
    "t_min": 420.0,
    "t_max": 600.0,
    "t_step": 5.0
  },
  "hill_climbing": {
    "max_sideways": 100,
    "max_restarts": 5
  },
  "simulated_annealing": {
    "initial_temp": 100.0,
    "cooling_rate": 0.995,
    "min_temp": 0.001,
    "max_iterations": 2500
  },
  "beam_search": {
    "beam_width": 5,
    "max_iterations": 150
  },
  "genetic_algorithm": {
    "population_size": 40,
    "generations": 120,
    "crossover_rate": 0.8,
    "mutation_rate": 0.25,
    "elitism": 2
  }
}
```

### Adding New Parameters
Solvers inherit from `LocalSearchSolver` and receive these dictionary parameters dynamically via `**kwargs`. To add a new parameter (e.g., `tournament_size`):
1. Add the key-value pair under the corresponding block in `config.json`.
2. Retrieve it inside your solver's `__init__` via `self.params.get("tournament_size", default_val)`.
3. No changes to `main.py` or the evaluation pipeline are needed.

---

## 8. Implementation Tasks

You must complete the implementation of files located in `algorithms/`:

1. **`algorithms/operators.py`**:
   * `get_neighbors(state, problem)`: Generate adjacent configurations by perturbing departure times ($\pm \Delta t$) and finding valid, loop-free detour paths between waypoints.
   * `crossover(parent1, parent2, problem)`: Recombine two routes using a shared intermediate junction node and blend their departure times.
2. **`algorithms/hill_climbing.py`**: Implement steepest descent with sideways move tracking and restarts.
3. **`algorithms/simulated_annealing.py`**: Implement the Metropolis acceptance criterion and temperature cooling schedule.
4. **`algorithms/beam_search.py`**: Implement pooled successor generation across $K$ states and select the top $K$.
5. **`algorithms/genetic_algorithm.py`**: Implement fitness calculation, parent selection, crossover, mutation, and elitism.

### Invariants & Requirements
* **Path Validity:** Every generated path must be a continuous sequence of directed edges from `problem.origin` to `problem.destination`.
* **No Cycles:** Paths must not contain duplicate nodes.
* **Time Bounds:** Departure times must remain within $[t_{\min}, t_{\max}]$.
* **History Tracking:** Record the minimum objective cost at each step into `self.history` for metric plotting.

---

## 9. Final Report & Submission Guidelines

### 9.1 Repository Submission
1. Fork this repository to your personal GitHub account.
2. Ensure you commit your progress incrementally. Your commit log should reflect systematic development across all four algorithms.
3. Submit the link to your forked GitHub repository on the course platform (Quera).
4. **Deadline Policy:** Push all commits prior to the deadline. Post-deadline commits are subject to severe penalties or outright disqualification.

---

### 9.2 Comprehensive Report Deliverable (`REPORT.md` or `report.pdf`)

Alongside your codebase, you must include a detailed technical report in the root directory of your repository (written in either English or Persian).

Your report must cover the following core sections:
* **Architecture & Operator Design:** Explain how your neighborhood generator balances time tweaks versus route detours. How do you guarantee that mutated and crossover paths remain continuous, valid, and free of topological cycles?
* **Empirical Comparison Table:** A comprehensive summary table comparing all four algorithms across:
  * Best objective value achieved ($f(x^*)$ in minutes).
  * Best departure time ($t^*$) and optimal path string.
  * Average execution runtime (in seconds).
  * Iteration count until convergence.
* **Visual Convergence Analysis:** Include the generated `convergence_comparison.png` plot. Analyze the distinct search profiles of each solver (e.g., greedy descents, stochastic hill escapes, beam diversity, GA generation fitness progression).
* **Hyperparameter Sensitivity:** Detail your experiments with tuning `config.json`. What parameters caused premature convergence or thrashing, and what configurations yielded optimal stability?

---

### 9.3 Analytical Questions (Mandatory to Address in Your Report)

You must include explicit answers to at least 5 of the following questions in your report (arbitrary; we'll score the best 5 of your responses):

#### Question 1: Continuous Optimization & Gradient Descent
In continuous optimization problems (such as the airport siting problem from the course slides), Gradient Descent updates states using:

$$x \leftarrow x - \alpha \nabla f(x)$$

* Why is classical Gradient Descent fundamentally inapplicable to optimizing the path $P$ in this road network problem?
* Could Gradient Descent be used to optimize the departure time $t_{\text{start}}$ in isolation while holding path $P$ fixed? What mathematical conditions would the dynamic congestion multiplier $C_e(t)$ need to satisfy for this to be valid?

#### Question 2: Local Search vs. Classical Incremental Search
Classical graph search algorithms (e.g., A*, Dijkstra) construct routes edge-by-edge.
* In our problem, why is the commute planned as a *complete configuration* $x = \langle P, t_{\text{start}} \rangle$ rather than using incremental state-space search?
* Under what network conditions or constraints would incremental time-dependent Dijkstra struggle compared to local search heuristics?

#### Question 3: Local Beam Search vs. Genetic Algorithm Recombination
The course slides describe Local Beam Search as maintaining $K$ parallel searches and ask: *"What other well-known algorithm does this remind you of? Evolution!"*
* What fundamental advantage does the **crossover** operator in Genetic Algorithms offer over the pooled greedy selection mechanism in Local Beam Search?
* Did your Local Beam Search suffer from the classic issue where all $K$ beams quickly converge onto the same single local peak? How did you or could you prevent this population collapse?

#### Question 4: Plateaus and Sideways Moves in Hill Climbing
On the 8-Queens problem, allowing up to 100 sideways moves increased the success probability from $14\%$ to $94\%$.
* In our commute network, what concrete scenario creates a "shoulder" or "flat local minimum" where neighboring configurations yield identical or nearly identical travel times?
* How did your choice of `max_sideways` impact the runtime versus solution quality in `HillClimbingSolver`?

#### Question 5: Cooling Schedule Dynamics in Simulated Annealing
Simulated Annealing relies on the Metropolis acceptance probability:

$$P = \exp\left(-\frac{\Delta E}{T}\right)$$

* What are the risks of setting the initial temperature $T_0$ too high versus too low?
* What happens if the cooling rate $\alpha$ is too small (e.g., $\alpha = 0.5$, rapid quenching) versus very close to $1.0$ (e.g., $\alpha = 0.999$)? Provide concrete observations from your experiments.

#### Question 6: Crossover Disjointness & Epistasis
Consider two parent solutions $x_1 = (P_1, t_1)$ and $x_2 = (P_2, t_2)$ that traverse completely disjoint corridors of the city—sharing no intermediate nodes between `Home` and `Uni`.
* How does your `crossover()` function handle this edge case without generating invalid paths or broken topological edges?
* Why can crossing over two individually high-quality parent routes sometimes produce a catastrophically poor offspring (high epistasis)?

#### Question 7: Combinatorial Space Estimation
Suppose a road network has $|V|$ nodes, an average node out-degree of $b$, an average origin-to-destination path length of $L$ hops, and a departure time window containing $N_t$ discrete intervals.
* Formulate a theoretical upper bound for the total number of complete states $|\mathcal{X}|$.
* Based on your estimate, explain why exhaustive brute-force search is intractable on large road networks, justifying the use of local search heuristics.
