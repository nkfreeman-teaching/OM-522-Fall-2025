import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Shifting Bottleneck Heuristic for Job Shop Scheduling

    This notebook implements the **Shifting Bottleneck Heuristic (SBN)**, a powerful algorithm for solving job shop scheduling problems.

    ## What is the Shifting Bottleneck Heuristic?

    The SBN heuristic is an iterative algorithm that:
    1. Identifies the machine that is currently the "bottleneck" (most constrained)
    2. Optimally schedules that machine using single-machine scheduling techniques
    3. Fixes that machine's sequence and repeats for the next bottleneck

    ## Problem Context

    **Job Shop Scheduling**: Given a set of jobs, each requiring processing on multiple machines in a specific order, the goal is to minimize the total completion time (makespan) while respecting:
    - Machine availability (one job at a time per machine)
    - Job routing (each job must visit machines in a prescribed sequence)
    - Processing time requirements

    ## Algorithm Approach

    The SBN heuristic uses:
    - **Critical Path Method (CPM)** to analyze the current schedule network
    - **Single-machine scheduling** to minimize maximum lateness for each candidate bottleneck
    - **Iterative bottleneck selection** based on maximum lateness values
    """
    )
    return


@app.cell
def _():
    import polars as pl

    import sbn_utilities
    return pl, sbn_utilities


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Loading and Preprocessing

    The input data (`SBN_data.csv`) contains job shop scheduling information with the following format:
    - **job**: Job identifier
    - **machine_sequence**: Comma-separated list of machines the job must visit in order
    - **pij**: Comma-separated list of processing times for each machine operation

    We transform this into a **CPM (Critical Path Method) network** format where:
    - Each node represents a specific operation: `machine,job` (e.g., "3,2" means Job 2 on Machine 3)
    - Edges represent precedence constraints (job routing + machine sequencing)
    - Each node has a duration (`pij`)
    """
    )
    return


@app.cell
def _(pl):
    # Load raw job shop data from CSV and parse comma-separated fields
    raw_data = pl.read_csv(
        'data/SBN_data.csv'
    ).with_columns(
        pl.col('machine_sequence').str.split(','),
        pl.col('pij').str.split(',')
    )

    # Extract list of all machines and initialize tracking variables
    machine_list = raw_data['machine_sequence'].explode().unique().sort().to_list()
    scheduled_machines = []

    # Convert job shop data into CPM network format
    # Each operation becomes a node with format "machine,job"
    cpm_data = []
    for row in raw_data.to_dicts():
        current_job = row.get('job')
        previous_machine_job = None
        row_machine_sequence = row.get('machine_sequence')
        row_pij = row.get('pij')

        # For each operation in the job's routing
        for machine, pij in zip(row_machine_sequence, row_pij):
            current_machine_job = f'{machine},{current_job}'

            # Add precedence constraint from previous operation in the same job
            if previous_machine_job is None:
                predecessors = None  # First operation has no job predecessor
            else:
                predecessors = [previous_machine_job]

            cpm_data.append({
                'current_machine_job': current_machine_job,
                'predecessors': predecessors,
                'pij': pij,
            })
            previous_machine_job = current_machine_job

    # Convert to DataFrame and ensure processing times are integers
    cpm_data = pl.DataFrame(
        cpm_data
    ).with_columns(
        pl.col('pij').cast(pl.Int64)
    )
    return cpm_data, machine_list, scheduled_machines


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Key Data Structures

    The preprocessing creates three important variables:
    - **`cpm_data`**: Polars DataFrame representing the CPM network with columns:
      - `current_machine_job`: Operation identifier ("machine,job" format)
      - `predecessors`: List of predecessor operations (job routing constraints)
      - `pij`: Processing time for the operation
    - **`machine_list`**: List of all unique machines in the problem
    - **`scheduled_machines`**: Initially empty; tracks which machines have been scheduled
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Main Shifting Bottleneck Algorithm

    The algorithm iteratively schedules machines until all are scheduled:

    ### For each iteration:
    1. **Run CPM** on the current network to get early/late start/finish times
    2. **For each unscheduled machine**:
       - Extract jobs assigned to that machine
       - Formulate single-machine problem with release times (rj = early start) and due dates (dj = late finish)
       - Solve to minimize maximum lateness using EDD (Earliest Due Date) rule
    3. **Select bottleneck**: Machine with highest maximum lateness (Lmax)
    4. **Fix sequence**: Add precedence constraints between consecutive jobs on the bottleneck machine
    5. **Repeat** until all machines are scheduled
    """
    )
    return


@app.cell
def _(cpm_data, machine_list, pl, sbn_utilities, scheduled_machines):
    # Create local working copies (marimo scoping: use _ prefix for cell-local variables)
    _cpm_data = cpm_data.clone()
    _scheduled_machines = list(scheduled_machines)

    # Iterate until all machines have been scheduled
    while set(machine_list) - set(_scheduled_machines):
        _unscheduled_machines = list(set(machine_list) - set(_scheduled_machines))

        # Step 1: Run CPM analysis on current network
        _cpm_results = sbn_utilities.calculate_cpm(_cpm_data)

        # Step 2: Evaluate each unscheduled machine as potential bottleneck
        _all_single_machine_results = []
        for _current_machine in _unscheduled_machines:
            # Extract all jobs assigned to this machine
            _relevant_jobs = [
                job for job in _cpm_results.keys()
                if job.startswith(_current_machine)
            ]

            # Formulate single-machine scheduling problem
            # rj (release time) = early start from CPM
            # dj (due date) = late finish from CPM
            _current_single_machine_data = []
            for _job in _relevant_jobs:
                _early_start = _cpm_results[_job].get('early_start')
                _early_finish = _cpm_results[_job].get('early_finish')
                _processing_time = _early_finish - _early_start
                _late_finish = _cpm_results[_job].get('late_finish')

                _current_single_machine_data.append({
                    'job': _job,
                    'pj': _processing_time,
                    'rj': _early_start,
                    'dj': _late_finish,
                })

            # Solve single-machine problem to minimize maximum lateness
            _current_single_machine_data = pl.DataFrame(_current_single_machine_data)
            _single_machine_results = sbn_utilities.minimize_maximum_lateness(_current_single_machine_data)
            _sequence = _single_machine_results.get('sequence')
            _max_lateness = _single_machine_results.get('max_lateness')

            _all_single_machine_results.append({
                'machine': _current_machine,
                'sequence': _sequence,
                'Lmax': _max_lateness,
            })

        # Step 3: Select machine with highest maximum lateness as bottleneck
        _all_single_machine_results = pl.DataFrame(
            _all_single_machine_results
        ).sort(
            'Lmax',
            descending=True
        )
        _bottleneck_information = _all_single_machine_results.to_dicts()[0]
        _bottleneck_machine = _bottleneck_information.get('machine')
        _bottleneck_sequence = _bottleneck_information.get('sequence')
        print(f'Identified bottleneck is {_bottleneck_machine}')

        # Step 4: Add precedence constraints for the bottleneck machine's sequence
        # Convert to dictionary format for easier manipulation of predecessor lists
        _cpm_data_dicts = {
            row['current_machine_job']: {
                'predecessors': row['predecessors'] if row['predecessors'] else [],
                'pij': row['pij']
            }
            for row in _cpm_data.to_dicts()
        }

        # Add precedence constraints between consecutive jobs in bottleneck sequence
        for _new_predecessor, _target in zip(_bottleneck_sequence[:-1], _bottleneck_sequence[1:]):
            _cpm_data_dicts[_target]['predecessors'].append(_new_predecessor)

        # Convert back to polars DataFrame
        _updated_rows = [
            {
                'current_machine_job': _machine_job,
                'predecessors': _data['predecessors'] if _data['predecessors'] else None,
                'pij': _data['pij']
            }
            for _machine_job, _data in _cpm_data_dicts.items()
        ]
        _cpm_data = pl.DataFrame(_updated_rows)

        # Mark this machine as scheduled
        _scheduled_machines.append(_bottleneck_machine)

    final_solution = _cpm_data.clone()
    return (final_solution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Results and Visualization

    After completing the shifting bottleneck heuristic, we have:
    - **final_solution**: A polars DataFrame containing the complete schedule with all precedence constraints
    - Each row represents an operation with its machine, job, predecessors, and processing time

    The Gantt chart below visualizes the schedule showing:
    - Each machine as a separate row
    - Jobs colored by job number
    - Time on the x-axis showing when each operation is scheduled
    - The schedule uses early start times from the final CPM calculation
    """
    )
    return


@app.cell
def _(final_solution, sbn_utilities):
    sbn_utilities.create_gantt_chart(final_solution)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
