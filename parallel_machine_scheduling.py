import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import pathlib
    import random

    import polars as pl
    from tqdm.auto import tqdm
    return mo, pathlib, pl, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Function Defintions""")
    return


@app.cell
def _(M, pj_values, pl):
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


    def get_schedule_details(schedule_dict: dict) -> dict:

        schedule_details = {}
        for _machine, _machine_sequence in schedule_dict.items():
            start_time = 0
            _machine_details = {}
            for _job in _machine_sequence:
                _machine_details[_job] = {
                    'start_time': start_time,
                    'completion_time': start_time + pj_values[_job],
                }
                start_time = start_time + pj_values[_job]
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
    return (
        compute_all_machine_workloads,
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
    data,
    get_lpt_schedule,
    get_schedule_details,
    make_gantt_chart,
    pj_values,
):
    LPT_schedule = get_lpt_schedule(
        data=data,
        pj_dict=pj_values,
    )

    LPT_schedule_details = get_schedule_details(LPT_schedule)

    make_gantt_chart(LPT_schedule_details)
    return (LPT_schedule,)


@app.cell
def _(LPT_schedule, compute_all_machine_workloads, pj_values, random):
    incumbent = dict(LPT_schedule)

    incumbent_workloads = compute_all_machine_workloads(
        machine_assignments=incumbent,
        pj_dict=pj_values,
    )

    _max_workload_machine = max(
        incumbent_workloads.items(),
        key=lambda x: x[1]
    )[0]
    _max_workload_machine_jobs = incumbent[_max_workload_machine]

    print(f'{_max_workload_machine_jobs = }')

    _idx = random.randint(a=0, b=len(_max_workload_machine_jobs)-1)
    _selected_job = _max_workload_machine_jobs[_idx]

    _new_max_workload_machine_jobs = _max_workload_machine_jobs[:_idx] + _max_workload_machine_jobs[_idx+1:]
    print(f'{_new_max_workload_machine_jobs = }')


    _min_workload_machine = min(
        incumbent_workloads.items(),
        key=lambda x: x[1]
    )[0]
    _min_workload_machine_jobs = incumbent[_min_workload_machine]

    print(f'{_min_workload_machine_jobs = }')

    _idx = random.randint(a=0, b=len(_min_workload_machine_jobs)-1)

    _new_min_workload_machine_jobs = (
        _min_workload_machine_jobs[:_idx] 
        + [_selected_job]
        +_min_workload_machine_jobs[_idx:]
    )
    print(f'{_new_min_workload_machine_jobs = }')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
