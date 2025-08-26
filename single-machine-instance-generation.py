import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import polars as pl
    return np, pl


@app.cell
def _(np, pl):
    np.random.seed(42)

    n = 10

    pj_low = 5
    pj_high = 15

    pj_array = np.random.randint(
        low=pj_low,
        high=pj_high,
        size=n
    )

    rj_array = np.random.randint(
        low=0,
        high=(n/2)*((pj_low + pj_high)/2),
        size=n
    )

    dj_offset = np.random.randint(
        low=0,
        high=25,
        size=n
    )

    dj_array = rj_array + pj_array + dj_offset

    wj_array = np.random.randint(
        low=1,
        high=10,
        size=n
    )

    data = pl.DataFrame({
        'j': [i for i in range(1, n + 1)],
        'pj': pj_array,
        'rj': rj_array,
        'dj': dj_array,
        'wj': wj_array,
    })

    data.write_csv('test1.csv')
    return (data,)


@app.cell
def _(data, pl):
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
            selected_job = selected_job_data.get('j')

            schedule.append({'j': selected_job, 't': t, 'pj': selected_job_pj, 'Cj': t + selected_job_pj})
            scheduled_jobs.add(selected_job)
            unscheduled_jobs = unscheduled_jobs - scheduled_jobs
            t += selected_job_pj
        else:
            t = data.filter(
                pl.col('j').is_in(unscheduled_jobs)
            ).get_column(
                'rj'
            ).min()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
