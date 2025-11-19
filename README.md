# OM-522-Fall-2025

**Operations Management Scheduling Algorithms** - Fall 2025 offering of OM 522

This repository contains Python implementations of classical scheduling algorithms and optimization techniques for various operational research problems, including job shop scheduling, single machine scheduling, parallel machine scheduling, project scheduling (CPM), and the Traveling Salesman Problem (TSP).

## Table of Contents

- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Main Algorithms](#main-algorithms)
- [Data Formats](#data-formats)
- [Development Guide](#development-guide)
- [Additional Resources](#additional-resources)

## Quick Start

### Installation

This project uses [Pixi](https://prefix.dev/) for package management. Install dependencies:

```bash
pixi install
```

### Running Notebooks

All main algorithms are implemented as interactive [Marimo](https://marimo.io/) notebooks:

```bash
# Run interactively (recommended)
pixi run marimo edit shifting_bottleneck_heuristic.py

# Or run as script
pixi run python shifting_bottleneck_heuristic.py
```

## Repository Structure

```
OM-522-Fall-2025/
├── data/                          # Input data files
│   ├── SBN_data.csv              # Job shop scheduling data
│   ├── project_data.csv          # CPM project data
│   └── tsp_AL_100.csv            # TSP Alabama cities coordinates
├── test_instances_20250930/      # Test instances for experiments
├── utilities/                     # Utility scripts for data generation
├── sbn_utilities.py              # Shared utility functions (CPM, EDD, Gantt charts)
└── [Main algorithm notebooks]    # Interactive Marimo notebooks
```

### Main Algorithm Files

| File | Algorithm | Objective |
|------|-----------|-----------|
| `shifting_bottleneck_heuristic.py` | Shifting Bottleneck | Minimize makespan (job shop) |
| `parallel_machine_scheduling.py` | LPT + Neighborhood Search | Minimize makespan (parallel machines) |
| `neighborhood_search_min_wjtj.py` | SPT + Neighborhood Search | Minimize Σ wⱼTⱼ (weighted tardiness) |
| `neighborhood_search_TSP.py` | Nearest Neighbor + SSR | Minimize tour distance (TSP) |
| `critical_path_method.py` | Critical Path Method | Project scheduling analysis |
| `clarke-wright-savings.py` | Clarke-Wright Savings | Minimize distance (vehicle routing) |
| `critical_ratio_dispatching_rule.py` | Critical Ratio Rule | Single machine dispatching |

## Main Algorithms

### 1. Shifting Bottleneck Heuristic (Job Shop Scheduling)

**File**: `shifting_bottleneck_heuristic.py`

Solves the job shop scheduling problem to minimize makespan using an iterative bottleneck identification approach.

**How it works**:
1. Converts job shop data into a CPM network format
2. Iteratively identifies the machine with highest maximum lateness (the "bottleneck")
3. Optimally sequences jobs on that machine using single-machine scheduling (EDD rule)
4. Fixes the sequence and repeats for remaining machines

**Input**: `data/SBN_data.csv` with columns `job`, `machine_sequence`, `pij`

**Quick run**:
```bash
pixi run marimo edit shifting_bottleneck_heuristic.py
```

### 2. Parallel Machine Scheduling

**File**: `parallel_machine_scheduling.py`

Minimizes makespan when scheduling jobs on M identical parallel machines using LPT heuristic with neighborhood search improvement.

**How it works**:
1. Initial solution: Longest Processing Time (LPT) first
2. Improvement: Insertion neighborhood (move job from max-workload to min-workload machine)
3. Stops after 5,000,000 non-improving iterations

**Input**: Test instances in `test_instances_20250930/` directory

**Quick run**:
```bash
pixi run marimo edit parallel_machine_scheduling.py
```

### 3. Single Machine Weighted Tardiness

**File**: `neighborhood_search_min_wjtj.py`

Minimizes total weighted tardiness (Σ wⱼTⱼ) for single machine scheduling with release times and due dates.

**How it works**:
1. Initial solution: Shortest Processing Time (SPT) with release times
2. Improvement: Neighborhood search with two operators:
   - **API**: Adjacent Pairwise Interchange
   - **PI**: Random Pairwise Interchange
3. Compares both operators across multiple test instances

**Input**: Test instances in `test_instances_20250930/` directory

**Quick run**:
```bash
pixi run marimo edit neighborhood_search_min_wjtj.py
```

### 4. Traveling Salesman Problem

**File**: `neighborhood_search_TSP.py`

Finds the shortest tour visiting all cities exactly once using multi-start neighborhood search.

**How it works**:
1. Initial solution: Nearest Neighbor heuristic from each city
2. Improvement: SSR (String String Reversal) neighborhood - reverse tour segments
3. Tries all 100 cities as starting points and returns the best tour found

**Input**: `data/tsp_AL_100.csv` (100 Alabama cities with coordinates)

**Quick run**:
```bash
pixi run marimo edit neighborhood_search_TSP.py
```

### 5. Critical Path Method (Project Scheduling)

**File**: `critical_path_method.py`

Analyzes project schedules using CPM to identify critical path and calculate slack times.

**How it works**:
1. Forward pass: Calculate Early Start (ES) and Early Finish (EF)
2. Backward pass: Calculate Late Start (LS) and Late Finish (LF)
3. Compute slack and identify critical path (tasks with zero slack)

**Input**: `data/project_data.csv` with columns `j` (job), `pj` (duration), `pred` (predecessors)

**Quick run**:
```bash
pixi run marimo edit critical_path_method.py
```

### 6. Clarke-Wright Savings (Vehicle Routing)

**File**: `clarke-wright-savings.py`

Solves the Capacitated Vehicle Routing Problem (CVRP) using the Clarke-Wright Savings algorithm.

**How it works**:
1. Initialize: Each customer starts in their own route
2. Calculate savings for each customer pair: s(i,j) = d(depot,i) + d(j,depot) - d(i,j)
3. Merge routes in order of highest savings while respecting capacity constraints
4. Use Nearest Neighbor to sequence visits within each cluster

**Input**: `data/tsp_AL_100.csv` with Birmingham as depot

**Quick run**:
```bash
pixi run marimo edit clarke-wright-savings.py
```

### 7. Critical Ratio Dispatching Rule

**File**: `critical_ratio_dispatching_rule.py`

Implements the Critical Ratio dispatching rule for single machine scheduling with release times.

**How it works**:
1. At each decision point, calculate CR = (dⱼ - t) / pⱼ for available jobs
2. Select the job with smallest critical ratio (most urgent)
3. Continue until all jobs are scheduled

**Input**: `data/10_job_test.csv` with columns `j`, `pj`, `rj`, `dj`

**Quick run**:
```bash
pixi run marimo edit critical_ratio_dispatching_rule.py
```

## Data Formats

### Job Shop Data

```csv
job,machine_sequence,pij
1,"1,2,3","5,3,2"
2,"2,1,3","4,6,3"
```
- `job`: Job identifier
- `machine_sequence`: Comma-separated list of machines to visit in order
- `pij`: Comma-separated processing times for each operation

### Single Machine Data

```csv
j,pj,rj,dj,wj
J1,5,0,10,2
J2,3,2,15,1
```
- `j`: Job ID
- `pj`: Processing time
- `rj`: Release time
- `dj`: Due date
- `wj`: Weight (importance)

### CPM Project Data

```csv
j,pj,pred
1,5,""
2,3,"1"
3,4,"1,2"
```
- `j`: Task number
- `pj`: Task duration
- `pred`: Comma-separated predecessor tasks (empty for start tasks)

### TSP Data

```csv
city,lat,lng
Birmingham,33.5207,-86.8025
Montgomery,32.3668,-86.3000
```
- `city`: City name
- `lat`: Latitude
- `lng`: Longitude

## Development Guide

### Key Technologies

- **Python 3.13+**
- **Polars**: DataFrame operations (preferred over pandas for this project)
- **Marimo**: Interactive notebooks
- **Matplotlib/Seaborn**: Visualization
- **Pixi**: Package management

### Marimo Variable Scoping

Marimo notebooks have special scoping rules:

⚠️ **Variables can only be assigned in ONE cell** (unless prefixed with `_`)

```python
# ✅ CORRECT
@app.cell
def _():
    data = load_data()
    return data,

@app.cell
def _(data):
    _processed = process(data)  # Use _ for local variables
    return result,

# ❌ WRONG - causes scoping error
@app.cell
def _(data):
    data = process(data)  # ERROR: data already assigned in another cell
    return data,
```

**Best practices**:
- Use `_` prefix for loop variables, intermediate computations, and local copies
- Only return variables that other cells need to access
- See `CLAUDE.md` for detailed scoping rules

### Shared Utilities (`sbn_utilities.py`)

Common functions used across multiple notebooks:

- `calculate_cpm(df)`: Critical Path Method implementation
- `minimize_maximum_lateness(df)`: Single-machine EDD scheduling
- `create_gantt_chart(df, output_file)`: Gantt chart visualization
- `parse_machine_job(str)`: Parse "machine,job" format strings

### Running Tests

Generate test instances using utility scripts:

```bash
# Located in utilities/ directory
pixi run python utilities/single-machine-instance-generation.py
```

## Additional Resources

### Documentation

- **CLAUDE.md**: Detailed technical documentation for Claude Code
- **Code comments**: Each notebook has inline documentation
- **Markdown cells**: Notebooks include explanatory markdown throughout

### Course Context

This repository supports OM-522 (Operations Management) with implementations of:
- Classical scheduling heuristics (LPT, SPT, EDD)
- Neighborhood search techniques (API, PI, SSR)
- Project management tools (CPM, Gantt charts)
- Optimization algorithms (Shifting Bottleneck)

### Contributing

When modifying notebooks:
1. Maintain marimo variable scoping rules (use `_` prefixes)
2. Add markdown cells to explain complex sections
3. Use Polars for data operations (project standard)
4. Update CLAUDE.md if adding new patterns or utilities

---

**Author**: Nickolas K Freeman
**Course**: OM 522 - Fall 2025
**Environment**: Managed with Pixi
