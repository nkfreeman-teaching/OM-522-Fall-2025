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
    mo.md(r"""# Function Defintions""")
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
    mo.md(r"""# Data Preparation""")
    return


@app.cell
def _(pathlib, pl):
    M = 3

    data_filepath = pathlib.Path('test_instances_20250930/instance-41-4.csv')
    assert data_filepath.exists()

    data = pl.read_csv(
        data_filepath,
        columns=['j', 'pj']
    )
    pj_values = data.to_pandas().set_index('j')['pj'].to_dict()
    return M, data, pj_values


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
    max_ni_iterations = 5_000_000

    # construct initial solution
    incumbent_solution = get_lpt_schedule(
        data=data,
        pj_dict=pj_values,
    )
    incumbent_value = compute_makespan(
        machine_schedule=incumbent_solution,
        pj_dict=pj_values
    )

    # execute neighborhood search
    ns_data = []
    count = 0
    ni_iterations = 0
    while ni_iterations < max_ni_iterations:
        ni_iterations += 1
        count += 1
    
        neighbor_solution = generate_insertion_neighbor(
            incumbent_solution=incumbent_solution,
            pj_dict=pj_values,
        )
        neighbor_value = compute_makespan(
            machine_schedule=neighbor_solution,
            pj_dict=pj_values
        )
        if neighbor_value < incumbent_value:
            ni_iterations = 0
            incumbent_solution = dict(neighbor_solution)
            incumbent_value = neighbor_value
        ns_data.append({
            'iteration': count,
            'incumbent_value': incumbent_value,
        })
    

    incumbent_schedule_details = get_schedule_details(
        schedule_dict=incumbent_solution,
        pj_dict=pj_values,
    )

    make_gantt_chart(incumbent_schedule_details)
    return (ns_data,)


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
