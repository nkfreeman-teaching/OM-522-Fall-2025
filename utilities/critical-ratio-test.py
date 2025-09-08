import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib

    import numpy as np
    import polars as pl
    return pathlib, pl


@app.cell
def _(pathlib, pl):
    data_filepath = pathlib.Path('data/10_job_test.csv')
    assert data_filepath.exists()

    data = pl.read_csv(data_filepath)

    unscheduled_jobs = set(data['j'].to_list())
    scheduled_jobs = set()
    schedule = []
    t = 0

    while unscheduled_jobs:
        available_jobs = data.filter(
            pl.col('j').is_in(unscheduled_jobs)
        ).filter(
            pl.col('rj') <= t
        ).get_column(
            'j'
        ).to_list()
        if available_jobs:
            available_job_data = data.filter(
                pl.col('j').is_in(available_jobs)
            ).with_columns(
                CR = (pl.col('dj') - t)/pl.col('pj')
            ).sort(
                'CR'
            )

            selected_job_data = available_job_data.to_dicts()[0]
            selected_job_pj = selected_job_data.get('pj')
            selected_job_rj = selected_job_data.get('rj')
            selected_job_dj = selected_job_data.get('dj')
            selected_job_wj = selected_job_data.get('wj')
            selected_job = selected_job_data.get('j')

            selected_job_Cj = t + selected_job_pj
            selected_job_lateness = selected_job_Cj - selected_job_dj
            selected_job_tardiness = max(selected_job_Cj - selected_job_dj, 0)

            schedule.append({
                'j': selected_job, 
                't': t, 
                'pj': selected_job_pj,
                'rj': selected_job_rj,
                'dj': selected_job_dj,
                'wj': selected_job_wj,
                'Cj': selected_job_Cj,
                'Lj': selected_job_lateness,
                'Tj': selected_job_tardiness,
                'wjTj': selected_job_wj*selected_job_tardiness,
            })
            scheduled_jobs.add(selected_job)
            unscheduled_jobs = unscheduled_jobs - scheduled_jobs
            t += selected_job_pj
        else:
            t = data.filter(
                pl.col('j').is_in(unscheduled_jobs)
            ).get_column(
                'rj'
            ).min()

    schedule = pl.DataFrame(
        schedule,
    )
    with pl.Config(tbl_width_chars=150, tbl_cols=10):
        print(schedule)

    sum_Cj = schedule['Cj'].sum()
    sum_wjTj = schedule['wjTj'].sum()
    Lmax = schedule['Lj'].max()

    print(f' - {sum_Cj = :,}')
    print(f' - {sum_wjTj = :,}')
    print(f' - {Lmax = :,}')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
