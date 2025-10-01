import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib
    import random

    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    return pathlib, pl, random


@app.cell
def _(pl, random):
    def get_SPT_solution(data: pl.DataFrame) -> list:

        all_jobs = data['j'].to_list()
    
        unscheduled_jobs = set(all_jobs)
        schedule = []
    
        t = data['rj'].min()
    
        while unscheduled_jobs:
    
            available_job_data = data.filter(
                pl.col('rj') <= t
            ).filter(
                pl.col('j').is_in(unscheduled_jobs)
            ).sort(
                'pj'
            )
            if len(available_job_data) > 0:
                selected_job = available_job_data.item(
                    row=0, 
                    column='j',
                )
                selected_job_pj = available_job_data.item(
                    row=0, 
                    column='pj',
                )
                schedule.append(selected_job)
                unscheduled_jobs = set(all_jobs) - set(schedule)
            
                t += selected_job_pj
            else:
                t = data.filter(
                    pl.col('j').is_in(unscheduled_jobs)
                )['rj'].min()
    
        return schedule


    def compute_weighted_tardiness(
        data: pl.DataFrame,
        solution: list,
    ) -> int:

        data_pd = data.to_pandas().set_index('j')
    
        t = data_pd['rj'].min()
        wjTj = 0
        for job in solution:
            if data_pd.loc[job, 'rj'] <= t:
                completion_time = t + data_pd.loc[job, 'pj']
            else:
                t = data_pd.loc[job, 'rj']
                completion_time = t + data_pd.loc[job, 'pj']
        
            if completion_time > data_pd.loc[job, 'dj']:
                tardiness = completion_time - data_pd.loc[job, 'dj']
            else:
                tardiness = 0
    
            wjTj += tardiness * data_pd.loc[job, 'wj']
            t = completion_time

        return wjTj


    def compute_API_neighbor(solution: list) -> list:

        index = random.randint(a=0, b=len(solution) -2)
        neighbor = list(solution)
        neighbor[index], neighbor[index+1] = neighbor[index+1], neighbor[index]

        return neighbor
    return compute_API_neighbor, compute_weighted_tardiness, get_SPT_solution


@app.cell
def _(compute_weighted_tardiness, get_SPT_solution, pathlib, pl):
    data_directory = pathlib.Path('test_instances_20250930/')
    assert data_directory.exists()

    data_filepaths = sorted(list(data_directory.glob('*.csv')))
    data_filepath = data_filepaths[0]

    data = pl.read_csv(data_filepath)
    spt_solution = get_SPT_solution(data)
    spt_solution_value = compute_weighted_tardiness(
        data=data,
        solution=spt_solution,
    )
    print(spt_solution_value)
    return data, spt_solution


@app.cell
def _(compute_API_neighbor, compute_weighted_tardiness, data, spt_solution):
    neighborhood_function = compute_API_neighbor
    max_non_improving_iterations = 1_000
    solution = list(spt_solution)
    objective_function = compute_weighted_tardiness

    incumbent = list(solution)
    incumbent_value = objective_function(
        data=data, 
        solution=incumbent,
    )
    non_improving_iterations = 0
    while non_improving_iterations < max_non_improving_iterations:
        non_improving_iterations += 1
    
        neighbor = neighborhood_function(incumbent)
        neighbor_value = objective_function(
            data=data, 
            solution=neighbor,
        )
        if neighbor_value < incumbent_value:
            incumbent = list(neighbor)
            incumbent_value = neighbor_value
            non_improving_iterations = 0
    return (neighbor_value,)


@app.cell
def _(neighbor_value):
    neighbor_value
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
