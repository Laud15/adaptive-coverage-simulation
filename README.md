# Adaptive Coverage Simulation

An agent-based simulation of adaptive drone coverage over points of interest. The project models fixed-wing drones and quadcopters in a continuous environment, with local perception, inter-drone communication, and real-time visualization of coverage and performance metrics.

## Technology stack

The exact dependency versions are recorded in `uv.lock`.

| Technology | Version |
| --- | --- |
| Python | 3.14 |
| uv | 0.10.2 |
| Mesa | 3.5.1 |

## Project structure

```text
adaptive-coverage-simulation/
|-- src/
|   |-- Agents.py   # Drone and point-of-interest agents
|   |-- Model.py    # Simulation model, scheduling, and data collection
|   `-- App.py      # Solara interface and real-time visualization
|-- pyproject.toml  # Project metadata and direct dependencies
|-- uv.lock         # Reproducible dependency lock file
`-- README.md
```

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available, then clone the repository:

```bash
git clone https://github.com/Laud15/adaptive-coverage-simulation.git
cd adaptive-coverage-simulation
```

Install Python 3.14 and synchronize the environment using the locked dependencies:

```bash
uv python install 3.14
uv sync --frozen
```

## Running the simulation

Start the Solara application from the project root:

```bash
uv run -m solara run src/App.py
```

Open the local address displayed in the terminal. The interface provides simulation controls, adjustable model parameters, a real-time map, and optional performance plots.
