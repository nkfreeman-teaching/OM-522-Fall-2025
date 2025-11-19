import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import pathlib
    import random

    import polars as pl
    import seaborn as sns
    from tqdm.auto import tqdm
    return mo, pathlib, pl, random, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Parallel Machine Scheduling with Neighborhood Search

    This notebook solves the **parallel machine scheduling problem** to minimize makespan (completion time of the last job).

    ## Problem Description

    Given:
    - A set of jobs, each with a processing time `pj`
    - M identical parallel machines

    **Objective**: Assign jobs to machines to minimize the **makespan** (the maximum completion time across all machines).

    The makespan equals the workload of the most heavily loaded machine.

    ## Solution Approach

    1. **Initial Solution**: Use Longest Processing Time (LPT) heuristic
       - Sort jobs by processing time (longest first)
       - Assign each job to the machine with minimum current workload

    2. **Improvement**: Use neighborhood search with insertion moves
       - Move a job from the most loaded machine to the least loaded machine
       - Accept if makespan improves
       - Continue until no improvement for a specified number of iterations
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Function Definitions""")
    return


@app.cell
def _(M, pj_values, pl, random):
    def compute_workload(
        job_list: list,
        pj_dict: dict,
    ) -> int:

        return sum([pj_dict.get(job) for job in job_list])


    def compute_all_machine_workloads(
        machine_assignments: dict,
        pj_dict: dict,
    ):

        current_workloads = {}
        for _current_machine, _current_assignments in machine_assignments.items():
            current_workloads[_current_machine] = compute_workload(
                job_list=_current_assignments,
                pj_dict=pj_dict,
            )

        return current_workloads


    def get_lpt_schedule(
        data: pl.DataFrame,
        pj_dict: dict,
    ) -> dict:

        LPT_order = data.sort(
            ['pj', 'j'],
            descending=[True, False]
        ).get_column(
            'j'
        ).to_list()

        LPT_machine_assignments = {}
        for idx in range(1, M+1):
            LPT_machine_assignments[f'M{idx}'] = []

        current_workloads = compute_all_machine_workloads(
            machine_assignments=LPT_machine_assignments,
            pj_dict=pj_values,
        )

        for _job in LPT_order:
            _selected_machine = min(
                current_workloads.items(),
                key=lambda x: x[1]
            )[0]
            LPT_machine_assignments[_selected_machine].append(_job)

            current_workloads = compute_all_machine_workloads(
                machine_assignments=LPT_machine_assignments,
                pj_dict=pj_values,
            )

        return LPT_machine_assignments


    def get_spt_schedule(
        data: pl.DataFrame,
        pj_dict: dict,
    ) -> dict:

        SPT_order = data.sort(
            ['pj', 'j'],
            descending=[False, False]
        ).get_column(
            'j'
        ).to_list()

        SPT_machine_assignments = {}
        for idx in range(1, M+1):
            SPT_machine_assignments[f'M{idx}'] = []

        current_workloads = compute_all_machine_workloads(
            machine_assignments=SPT_machine_assignments,
            pj_dict=pj_values,
        )

        for _job in SPT_order:
            _selected_machine = min(
                current_workloads.items(),
                key=lambda x: x[1]
            )[0]
            SPT_machine_assignments[_selected_machine].append(_job)

            current_workloads = compute_all_machine_workloads(
                machine_assignments=SPT_machine_assignments,
                pj_dict=pj_values,
            )

        return SPT_machine_assignments


    def get_schedule_details(
        schedule_dict: dict,
        pj_dict: dict,
    ) -> dict:

        schedule_details = {}
        for _machine, _machine_sequence in schedule_dict.items():
            start_time = 0
            _machine_details = {}
            for _job in _machine_sequence:
                _machine_details[_job] = {
                    'start_time': start_time,
                    'completion_time': start_time + pj_dict[_job],
                }
                start_time = start_time + pj_dict[_job]
            schedule_details[_machine] = _machine_details

        return schedule_details


    def make_gantt_chart(schedule_details_dict):

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 3))

        for machine in schedule_details_dict:
            for job in schedule_details_dict[machine]:
                start_time = schedule_details_dict[machine][job]['start_time']
                completion_time = schedule_details_dict[machine][job]['completion_time']
                length = completion_time - start_time
                ax.barh(
                    machine, 
                    length,
                    left=start_time, 
                    edgecolor='black',
                )
                ax.text(
                    start_time + length/2,
                    machine,
                    job,
                    va='center',
                    ha='center',
                    color='k',
                )
            ax.spines[['right', 'top']].set_visible(False)

        ax.set_xlabel('Time')
        ax.set_ylabel('Machines')
        ax.set_title('Gantt Chart')
        plt.show()


    def generate_insertion_neighbor(
        incumbent_solution: dict,
        pj_dict: dict,
    ) -> dict:

        incumbent = dict(incumbent_solution)

        incumbent_workloads = compute_all_machine_workloads(
            machine_assignments=incumbent,
            pj_dict=pj_values,
        )

        _max_workload_machine = max(
            incumbent_workloads.items(),
            key=lambda x: x[1]
        )[0]
        _max_workload_machine_jobs = incumbent[_max_workload_machine]

        _idx = random.randint(a=0, b=len(_max_workload_machine_jobs)-1)
        _selected_job = _max_workload_machine_jobs[_idx]

        _new_max_workload_machine_jobs = _max_workload_machine_jobs[:_idx] + _max_workload_machine_jobs[_idx+1:]


        _min_workload_machine = min(
            incumbent_workloads.items(),
            key=lambda x: x[1]
        )[0]
        _min_workload_machine_jobs = incumbent[_min_workload_machine]

        _idx = random.randint(a=0, b=len(_min_workload_machine_jobs)-1)

        _new_min_workload_machine_jobs = (
            _min_workload_machine_jobs[:_idx] 
            + [_selected_job]
            +_min_workload_machine_jobs[_idx:]
        )

        neighbor = dict(incumbent)
        neighbor[_max_workload_machine] = list(_new_max_workload_machine_jobs)
        neighbor[_min_workload_machine] = list(_new_min_workload_machine_jobs)

        return neighbor


    def compute_makespan(
        machine_schedule: dict,
        pj_dict: dict,
    ) -> int:

        _machine_workloads = compute_all_machine_workloads(
            machine_assignments=machine_schedule,
            pj_dict=pj_dict,
        )

        return max(_machine_workloads.values())
    return (
        compute_makespan,
        generate_insertion_neighbor,
        get_lpt_schedule,
        get_schedule_details,
        make_gantt_chart,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Preparation

    We load a test instance from the `test_instances_20250930/` directory. Each instance contains:
    - **`j`**: Job identifier
    - **`pj`**: Processing time for the job

    We configure the number of parallel machines (`M`) and create a dictionary for fast job lookup.
    """
    )
    return


@app.cell
def _(pathlib, pl):
    M = 3

    _data_filepath = pathlib.Path('test_instances_20250930/instance-41-4.csv')
    assert _data_filepath.exists()

    data = pl.read_csv(
        _data_filepath,
        columns=['j', 'pj']
    )
    pj_values = data.to_pandas().set_index('j')['pj'].to_dict()
    return M, data, pj_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Neighborhood Search Execution

    The algorithm proceeds as follows:

    1. **Initialize**: Start with LPT solution
    2. **Search Loop**: Repeatedly generate neighbor solutions by moving a job from the max-workload machine to the min-workload machine
    3. **Acceptance**: Accept neighbor if it improves (reduces) the makespan
    4. **Stopping Criterion**: Stop after 5,000,000 non-improving iterations

    The search tracks the incumbent solution value at each iteration to visualize convergence.
    """
    )
    return


@app.cell
def _(
    compute_makespan,
    data,
    generate_insertion_neighbor,
    get_lpt_schedule,
    get_schedule_details,
    make_gantt_chart,
    pj_values,
):
    # All variables in this cell are local (marimo scoping)
    _max_ni_iterations = 100_000

    # construct initial solution
    _incumbent_solution = get_lpt_schedule(
        data=data,
        pj_dict=pj_values,
    )
    _incumbent_value = compute_makespan(
        machine_schedule=_incumbent_solution,
        pj_dict=pj_values
    )

    # execute neighborhood search
    ns_data = []
    _count = 0
    _ni_iterations = 0
    while _ni_iterations < _max_ni_iterations:
        _ni_iterations += 1
        _count += 1

        _neighbor_solution = generate_insertion_neighbor(
            incumbent_solution=_incumbent_solution,
            pj_dict=pj_values,
        )
        _neighbor_value = compute_makespan(
            machine_schedule=_neighbor_solution,
            pj_dict=pj_values
        )
        if _neighbor_value < _incumbent_value:
            _ni_iterations = 0
            _incumbent_solution = dict(_neighbor_solution)
            _incumbent_value = _neighbor_value
        ns_data.append({
            'iteration': _count,
            'incumbent_value': _incumbent_value,
        })


    _incumbent_schedule_details = get_schedule_details(
        schedule_dict=_incumbent_solution,
        pj_dict=pj_values,
    )

    make_gantt_chart(_incumbent_schedule_details)
    return (ns_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Convergence Visualization

    The plot below shows how the incumbent solution value (makespan) improves over iterations.
    Flat regions indicate periods where no improving neighbors were found.
    """
    )
    return


@app.cell
def _(ns_data, pl, sns):
    sns.lineplot(
        pl.DataFrame(ns_data),
        x='iteration',
        y='incumbent_value',
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
