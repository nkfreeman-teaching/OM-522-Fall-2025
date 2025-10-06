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
    from tqdm.auto import tqdm
    return pathlib, pl, plt, random, sns, tqdm


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
        data_dict,
        solution: list,
    ) -> int:

        t = 0
    
        wjTj = 0
        for job in solution:
            if data_dict[job]['rj'] <= t:
                completion_time = t + data_dict[job]['pj']
            else:
                t = data_dict[job]['rj']
                completion_time = t + data_dict[job]['pj']

            if completion_time > data_dict[job]['dj']:
                tardiness = completion_time - data_dict[job]['dj']
            else:
                tardiness = 0

            wjTj += tardiness * data_dict[job]['wj']
            t = completion_time

        return wjTj


    def compute_API_neighbor(solution: list) -> list:

        index = random.randint(a=0, b=len(solution) -2)
        neighbor = list(solution)
        neighbor[index], neighbor[index+1] = neighbor[index+1], neighbor[index]

        return neighbor


    def compute_PI_neighbor(solution: list) -> list:

        index1 = random.randint(a=0, b=len(solution) -1)
        index2 = random.randint(a=0, b=len(solution) -1)
        neighbor = list(solution)
        neighbor[index1], neighbor[index2] = neighbor[index2], neighbor[index1]

        return neighbor


    def run_neighborhood_search(
        neighborhood_function,
        solution: list,
        objective_function,
        data_dict,
        max_non_improving_iterations: int = 1_000,
    ) -> list:

        incumbent = list(solution)
        incumbent_value = objective_function(
            data_dict=data_dict, 
            solution=incumbent,
        )
        non_improving_iterations = 0
        while non_improving_iterations < max_non_improving_iterations:
            non_improving_iterations += 1

            neighbor = neighborhood_function(incumbent)
            neighbor_value = objective_function(
                data_dict=data_dict, 
                solution=neighbor,
            )
            if neighbor_value < incumbent_value:
                incumbent = list(neighbor)
                incumbent_value = neighbor_value
                non_improving_iterations = 0

        return incumbent
    return (
        compute_API_neighbor,
        compute_PI_neighbor,
        compute_weighted_tardiness,
        get_SPT_solution,
        run_neighborhood_search,
    )


@app.cell
def _(
    compute_API_neighbor,
    compute_PI_neighbor,
    compute_weighted_tardiness,
    get_SPT_solution,
    pathlib,
    pl,
    random,
    run_neighborhood_search,
    tqdm,
):
    data_directory = pathlib.Path('test_instances_20250930/')
    assert data_directory.exists()

    possible_neighborhood_functions = {
        'API': compute_API_neighbor,
        'PI': compute_PI_neighbor,
    }

    experiment_results = []

    data_filepaths = sorted(list(data_directory.glob('*.csv')))
    for data_filepath in tqdm(data_filepaths):
        for _neigborhood_function_name, _neigborhood_function in possible_neighborhood_functions.items():
            for _max_ni_iterations in [500, 1_000]:

                random.seed(0)

                data = pl.read_csv(data_filepath)
                data_pd = data.to_pandas().set_index('j')
                data_dict = data_pd.to_dict(orient='index')
            
                spt_solution = get_SPT_solution(data)
                spt_solution_value = compute_weighted_tardiness(
                    data_dict=data_dict,
                    solution=spt_solution,
                )

                best_neighbor_solution = run_neighborhood_search(
                    neighborhood_function=_neigborhood_function,
                    solution=spt_solution,
                    objective_function=compute_weighted_tardiness,
                    data_dict=data_dict,
                    max_non_improving_iterations=_max_ni_iterations,
                )
                best_neighbor_solution_value = compute_weighted_tardiness(
                    data_dict=data_dict,
                    solution=best_neighbor_solution,
                )

                experiment_results.append({
                    'filename': data_filepath.stem,
                    'neighborhood': _neigborhood_function_name,
                    'ni_iterations': _max_ni_iterations,
                    'best_value': best_neighbor_solution_value,
                })
    return (experiment_results,)


@app.cell
def _(experiment_results, pl):
    experiment_results_df = pl.DataFrame(experiment_results)
    return (experiment_results_df,)


@app.cell
def _(experiment_results_df, plt, sns):
    _fig, _ax = plt.subplots(1, 1, figsize=(6, 4))

    sns.pointplot(
        experiment_results_df,
        x='neighborhood',
        y='best_value',
        linestyle='none',
    )

    plt.show()
    return


@app.cell
def _(experiment_results_df, plt, sns):
    _fig, _ax = plt.subplots(1, 1, figsize=(6, 4))

    sns.pointplot(
        experiment_results_df,
        x='ni_iterations',
        y='best_value',
        linestyle='none',
    )

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
