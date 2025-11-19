import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pathlib

    import numpy as np
    import polars as pl
    return mo, pathlib, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Critical Ratio Dispatching Rule - Test

    This utility notebook tests the Critical Ratio dispatching rule implementation and calculates detailed performance metrics.

    ## Metrics Computed

    For each scheduled job:
    - **Cⱼ**: Completion time
    - **Lⱼ**: Lateness (Cⱼ - dⱼ)
    - **Tⱼ**: Tardiness (max(0, Lⱼ))
    - **wⱼTⱼ**: Weighted tardiness

    **Aggregate Metrics**:
    - Sum of completion times (Σ Cⱼ)
    - Sum of weighted tardiness (Σ wⱼTⱼ)
    - Maximum lateness (Lmax)
    """
    )
    return


@app.cell
def _(pathlib, pl):
    _data_filepath = pathlib.Path('data/10_job_test.csv')
    assert _data_filepath.exists()

    _data = pl.read_csv(_data_filepath)

    _unscheduled_jobs = set(_data['j'].to_list())
    _scheduled_jobs = set()
    _schedule = []
    _t = 0

    while _unscheduled_jobs:
        _available_jobs = _data.filter(
            pl.col('j').is_in(_unscheduled_jobs)
        ).filter(
            pl.col('rj') <= _t
        ).get_column(
            'j'
        ).to_list()
        if _available_jobs:
            _available_job_data = _data.filter(
                pl.col('j').is_in(_available_jobs)
            ).with_columns(
                CR = (pl.col('dj') - _t)/pl.col('pj')
            ).sort(
                'CR'
            )

            _selected_job_data = _available_job_data.to_dicts()[0]
            _selected_job_pj = _selected_job_data.get('pj')
            _selected_job_rj = _selected_job_data.get('rj')
            _selected_job_dj = _selected_job_data.get('dj')
            _selected_job_wj = _selected_job_data.get('wj')
            _selected_job = _selected_job_data.get('j')

            _selected_job_Cj = _t + _selected_job_pj
            _selected_job_lateness = _selected_job_Cj - _selected_job_dj
            _selected_job_tardiness = max(_selected_job_Cj - _selected_job_dj, 0)

            _schedule.append({
                'j': _selected_job,
                't': _t,
                'pj': _selected_job_pj,
                'rj': _selected_job_rj,
                'dj': _selected_job_dj,
                'wj': _selected_job_wj,
                'Cj': _selected_job_Cj,
                'Lj': _selected_job_lateness,
                'Tj': _selected_job_tardiness,
                'wjTj': _selected_job_wj*_selected_job_tardiness,
            })
            _scheduled_jobs.add(_selected_job)
            _unscheduled_jobs = _unscheduled_jobs - _scheduled_jobs
            _t += _selected_job_pj
        else:
            _t = _data.filter(
                pl.col('j').is_in(_unscheduled_jobs)
            ).get_column(
                'rj'
            ).min()

    _schedule = pl.DataFrame(
        _schedule,
    )
    with pl.Config(tbl_width_chars=150, tbl_cols=10):
        print(_schedule)

    _sum_Cj = _schedule['Cj'].sum()
    _sum_wjTj = _schedule['wjTj'].sum()
    _Lmax = _schedule['Lj'].max()

    print(f' - {_sum_Cj = :,}')
    print(f' - {_sum_wjTj = :,}')
    print(f' - {_Lmax = :,}')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
