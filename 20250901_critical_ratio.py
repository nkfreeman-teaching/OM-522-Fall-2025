import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    data = pl.read_csv('data/10_job_test.csv')

    t = 0
    unscheduled_jobs = set(data['j'].to_list())
    scheduled_jobs = set()
    schedule = []

    while unscheduled_jobs:
        available_job_data = data.filter(
            pl.col('j').is_in(unscheduled_jobs)
        ).filter(
            pl.col('rj') <= t
        )

        if len(available_job_data) > 0:
            selected_job_data = available_job_data.with_columns(
                CR = (pl.col('dj') - t)/pl.col('pj')
            ).sort(
                'CR'
            ).to_dicts()[0]

            selected_j = selected_job_data['j']
            selected_pj = selected_job_data['pj']
            selected_rj = selected_job_data['rj']
            selected_dj = selected_job_data['dj']

            print(f'At time {t}')
            print(f' - Schedule job {selected_j} which was released at {selected_rj} and due at {selected_dj}')
        
            schedule.append(selected_j)
            t = t + selected_pj

            scheduled_jobs.add(selected_j)
            unscheduled_jobs = unscheduled_jobs - scheduled_jobs
        else:
            min_rj = data.filter(
                pl.col('j').is_in(unscheduled_jobs)
            )['rj'].min()

            t = min_rj
    return (schedule,)


@app.cell
def _(schedule):
    schedule
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
