# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Python implementations for Operations Management scheduling algorithms and optimization techniques, specifically for the OM-522 Fall 2025 course. The codebase focuses on job shop scheduling, single machine scheduling, parallel machine scheduling, and project scheduling using various heuristics and optimization methods.

## Environment Setup

This project uses **Pixi** for package management (not pip/conda directly).

### Key Commands

```bash
# Install dependencies (if needed)
pixi install

# Run Python scripts
pixi run python <script.py>

# Run Marimo notebooks (interactive)
pixi run marimo edit <notebook.py>
```

### Dependencies

Python 3.13+ with core libraries:
- `polars` - DataFrame operations (preferred over pandas)
- `marimo` - Interactive notebook environment
- `matplotlib`, `seaborn` - Visualization
- `numpy`, `scikit-learn` - Numerical operations
- `tqdm` - Progress bars

## Code Architecture

### Core Modules

**`sbn_utilities.py`** - Shared utilities for scheduling algorithms
- `calculate_cpm(df)` - Critical Path Method implementation for project scheduling
  - Input: Polars DataFrame with columns: `current_machine_job`, `predecessors`, `pij` (duration)
  - Returns: Dictionary with early/late start/finish times for each task
  - Uses topological ordering for forward pass (early times) and reverse pass (late times)
- `minimize_maximum_lateness(df)` - Single-machine scheduling with Earliest Due Date (EDD) rule
  - Input: DataFrame with `job`, `pj` (processing time), `rj` (release time), `dj` (due date)
  - Returns: Dictionary with `sequence`, `max_lateness`, `job_details`
- `create_gantt_chart(df, output_file)` - Visualizes schedules using CPM results
- `parse_machine_job(str)` - Parses "machine,job" format strings (e.g., "3,2" → machine 3, job 2)

### Main Implementations

**Shifting Bottleneck Heuristic (SBN)** - `Shiting_Bottleneck_Heuristic.py`
- Iterative algorithm for job shop scheduling to minimize makespan
- Algorithm flow:
  1. Load job shop data from CSV (columns: `job`, `machine_sequence`, `pij`)
  2. Convert to CPM network format where nodes are "machine,job" operations
  3. Iteratively identify bottleneck machines:
     - Run CPM to get early/late times
     - For each unscheduled machine, formulate single-machine problem
     - Select machine with highest maximum lateness (Lmax)
     - Fix that machine's sequence by adding precedence constraints
  4. Repeat until all machines scheduled
- Uses `sbn_utilities.calculate_cpm()` and `sbn_utilities.minimize_maximum_lateness()`

**Parallel Machine Scheduling** - `parallel_machine_scheduling.py`
- Minimizes makespan using Longest Processing Time (LPT) heuristic + neighborhood search
- Key functions:
  - `get_lpt_schedule()` / `get_spt_schedule()` - Initial solutions
  - `generate_insertion_neighbor()` - Moves job from max-workload to min-workload machine
  - `compute_makespan()` - Objective function evaluation
- Non-improving iterations stopping criterion (default: 5,000,000)

**Single Machine Scheduling** - `ns_min_wjtj.py`
- Minimizes weighted tardiness (ΣwⱼTⱼ)
- Initial solution: SPT (Shortest Processing Time) with release times
- Neighborhood search with two operators:
  - `compute_API_neighbor()` - Adjacent Pairwise Interchange
  - `compute_PI_neighbor()` - Random Pairwise Interchange
- `run_neighborhood_search()` - Generic neighborhood search framework

**CPM Analysis** - `20251103_cpm_notebook.py`
- Standalone Critical Path Method implementation for project scheduling
- Identifies critical path (tasks with zero slack)
- Input: CSV with columns `j` (job), `pj` (duration), `pred` (comma-separated predecessors)
- Visualizes schedule with critical/non-critical tasks color-coded

## Data Format Conventions

### Job Shop Data (`data/SBN_data.csv`)
```
job,machine_sequence,pij
1,"1,2,3","5,3,2"
```
- `machine_sequence` - Comma-separated routing of machines
- `pij` - Comma-separated processing times for each operation

### Single Machine Data (`test_instances_20250930/*.csv`)
```
j,pj,rj,dj,wj
J1,5,0,10,2
```
- `j` - Job ID
- `pj` - Processing time
- `rj` - Release time
- `dj` - Due date
- `wj` - Weight for tardiness

### CPM Network Format (Internal)
```
current_machine_job,predecessors,pij
"3,2",["1,2","2,2"],5
```
- Operations represented as "machine,job"
- Predecessors is a list (or None for start operations)

## Marimo Variable Scoping Rules

Marimo has unique variable scoping rules that prevent common errors:

### Core Rules

1. **Single Assignment Across Cells**: A variable can only be assigned in ONE cell (unless prefixed with `_`)
   ```python
   # ❌ WRONG - causes scoping error
   @app.cell
   def _():
       data = load_data()  # Assigned here
       return data,

   @app.cell
   def _(data):
       data = process_data(data)  # ERROR: data assigned in two cells!
       return data,
   ```

   ```python
   # ✅ CORRECT - use underscore prefix for cell-local variables
   @app.cell
   def _():
       data = load_data()
       return data,

   @app.cell
   def _(data):
       _data = process_data(data)  # Local variable with _
       return _data,
   ```

2. **Underscore Prefix for Local Variables**: Variables starting with `_` are local to that cell
   ```python
   @app.cell
   def _(data):
       _temp = data.copy()
       _result = process(_temp)
       # _temp and _result are local to this cell
       return final_result,
   ```

3. **Function-Scoped Variables**: Variables inside functions are automatically local
   ```python
   @app.cell
   def _(pl):
       def process_data(df):
           temp = df.filter(...)  # temp is local to function
           return temp
       return process_data,
   ```

### Best Practices

- **Loop variables**: Always use `_` prefix for variables in loops/iterations
- **Temporary computations**: Use `_` prefix for intermediate values
- **Return only exports**: Only return variables that other cells need to access
- **Working copies**: When modifying data from another cell, create a local copy with `_` prefix

### Common Pattern
```python
@app.cell
def _(input_data):
    # Create local working copy
    _working_data = input_data.clone()

    # All loop/intermediate variables with underscore
    for _item in _working_data:
        _processed = process(_item)
        _results.append(_processed)

    # Return only what other cells need
    return final_output,
```

## Running the Code

### Marimo Notebooks
All main algorithms are implemented as Marimo notebooks (`.py` files with `marimo.App`):

```bash
# Interactive mode (recommended for development)
pixi run marimo edit Shiting_Bottleneck_Heuristic.py

# Run as script
pixi run python Shiting_Bottleneck_Heuristic.py
```

### Test Instances
- `data/` - Main input data files
- `test_instances_20250930/` - Single machine scheduling test instances
- Instance naming: `instance-<n>-<seed>.csv` where n is number of jobs

## Important Implementation Details

### Data Processing
- **Use Polars, not Pandas** for all new code (it's the project standard)
- CPM requires topological ordering - circular dependencies will raise `ValueError`
- Machine/job format: Always use "machine,job" string representation in SBN context

### Scheduling Algorithms
- **SBN**: Machines are scheduled one at a time, fixing sequences via precedence constraints
- **Neighborhood Search**: Uses non-improving iterations counter, not total iterations
- **CPM**: Forward pass calculates early times, backward pass uses reverse dependencies

### Visualization
- All Gantt charts use `matplotlib` with early start times from CPM
- Critical path tasks colored red, non-critical colored blue/lightblue
- Output saved to PNG files (e.g., `gantt_chart.png`)

## Common Patterns

### Converting Between Formats
```python
# Job shop → CPM network
for machine, pij in zip(machine_sequence, pij_list):
    current_machine_job = f'{machine},{job}'

# Polars → dict for iteration
data_dict = data.to_pandas().set_index('j').to_dict(orient='index')
```

### Neighborhood Search Template
```python
incumbent = initial_solution
incumbent_value = objective_function(incumbent)
ni_iterations = 0

while ni_iterations < max_ni_iterations:
    ni_iterations += 1
    neighbor = generate_neighbor(incumbent)
    neighbor_value = objective_function(neighbor)

    if neighbor_value < incumbent_value:
        incumbent = neighbor
        incumbent_value = neighbor_value
        ni_iterations = 0  # Reset counter
```

### CPM Usage
```python
cpm_results = sbn_utilities.calculate_cpm(df)
# Access results
early_start = cpm_results[task]['early_start']
late_finish = cpm_results[task]['late_finish']
slack = late_start - early_start
```
