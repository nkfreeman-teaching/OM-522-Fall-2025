import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Critical Ratio Dispatching Rule

    This notebook implements the **Critical Ratio (CR) dispatching rule** for single machine scheduling with release times.

    ## Problem Description

    Given a set of jobs where each job j has:
    - **pⱼ**: Processing time
    - **rⱼ**: Release time (earliest time the job can start)
    - **dⱼ**: Due date

    **Objective**: Create a schedule that prioritizes jobs based on their urgency relative to remaining processing time.

    ## Algorithm

    The Critical Ratio rule calculates at each decision point:

    **CR = (dⱼ - t) / pⱼ**

    Where:
    - dⱼ = due date of job j
    - t = current time
    - pⱼ = processing time of job j

    **Interpretation**:
    - CR < 1: Job is behind schedule (urgent)
    - CR = 1: Job is exactly on schedule
    - CR > 1: Job is ahead of schedule

    **Selection**: Choose the job with the **smallest CR** (most urgent).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Algorithm Implementation

    The algorithm proceeds as follows:
    1. Start at time t = 0
    2. At each decision point:
       - Find all jobs that have been released (rⱼ ≤ t)
       - Calculate CR for each available job
       - Select the job with minimum CR
       - Update time by adding the job's processing time
    3. If no jobs are available, advance time to the next release time
    4. Repeat until all jobs are scheduled
    """
    )
    return


@app.cell
def _(pl):
    _data = pl.read_csv('data/10_job_test.csv')

    _t = 0
    _unscheduled_jobs = set(_data['j'].to_list())
    _scheduled_jobs = set()
    schedule = []

    while _unscheduled_jobs:
        _available_job_data = _data.filter(
            pl.col('j').is_in(_unscheduled_jobs)
        ).filter(
            pl.col('rj') <= _t
        )

        if len(_available_job_data) > 0:
            _selected_job_data = _available_job_data.with_columns(
                CR = (pl.col('dj') - _t)/pl.col('pj')
            ).sort(
                'CR'
            ).to_dicts()[0]

            _selected_j = _selected_job_data['j']
            _selected_pj = _selected_job_data['pj']
            _selected_rj = _selected_job_data['rj']
            _selected_dj = _selected_job_data['dj']

            print(f'At time {_t}')
            print(f' - Schedule job {_selected_j} which was released at {_selected_rj} and due at {_selected_dj}')

            schedule.append(_selected_j)
            _t = _t + _selected_pj

            _scheduled_jobs.add(_selected_j)
            _unscheduled_jobs = _unscheduled_jobs - _scheduled_jobs
        else:
            _min_rj = _data.filter(
                pl.col('j').is_in(_unscheduled_jobs)
            )['rj'].min()

            _t = _min_rj
    return (schedule,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Results

    The final schedule shows the order in which jobs should be processed according to the Critical Ratio rule.
    """
    )
    return


@app.cell
def _(schedule):
    schedule
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
