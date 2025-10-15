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
    return mo, pathlib, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Function Defintions""")
    return


@app.function
def compute_workload(
    job_list: list,
    pj_dict: dict,
) -> int:

    return sum([pj_dict.get(job) for job in job_list])


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
def _(M, data):
    LPT_order = data.sort(
        ['pj', 'j'],
        descending=[True, False]
    ).get_column(
        'j'
    ).to_list()

    LPT_machine_assignments = {}
    for idx in range(1, M+1):
        LPT_machine_assignments[f'M{idx}'] = []
    return (LPT_machine_assignments,)


@app.cell
def _(LPT_machine_assignments):
    LPT_machine_assignments['M1'].extend([1, 2])
    LPT_machine_assignments['M2'].extend([10, 20])
    return


@app.cell
def _(LPT_machine_assignments, pj_values):
    current_workloads = {}
    for _current_machine, _current_assignments in LPT_machine_assignments.items():
        current_workloads[_current_machine] = compute_workload(
            job_list=_current_assignments,
            pj_dict=pj_values,
        )

    current_workloads
    return (current_workloads,)


@app.cell
def _(current_workloads):
    for x in current_workloads.items():
        print(f'{x[0] = }, {x[1] = }')
    return


@app.cell
def _(current_workloads):
    sorted(
        current_workloads.items(),
        key=lambda x: x[1]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
