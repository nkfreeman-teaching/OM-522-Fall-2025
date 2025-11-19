import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pathlib
    import random

    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    from tqdm.auto import tqdm
    return mo, pathlib, pl, plt, random, sns, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Single Machine Scheduling: Minimizing Weighted Tardiness

    This notebook solves the single machine scheduling problem with the objective of minimizing total weighted tardiness: **Σ wⱼTⱼ**

    ## Problem Description

    Given a set of jobs where each job j has:
    - **pⱼ**: Processing time
    - **rⱼ**: Release time (earliest time the job can start)
    - **dⱼ**: Due date
    - **wⱼ**: Weight (importance/priority)

    **Objective**: Find a sequence of jobs that minimizes the sum of weighted tardiness, where:
    - Tardiness Tⱼ = max(0, Cⱼ - dⱼ)  where Cⱼ is the completion time
    - Weighted tardiness = wⱼ × Tⱼ

    ## Solution Approach

    1. **Initial Solution**: Shortest Processing Time (SPT) with release times
       - Select available jobs (released) and process the one with shortest pⱼ
       - Good baseline for minimizing completion times

    2. **Improvement**: Neighborhood search with two operators:
       - **API (Adjacent Pairwise Interchange)**: Swap two adjacent jobs in the sequence
       - **PI (Pairwise Interchange)**: Swap any two random jobs in the sequence

    3. **Experimental Design**: Compare both neighborhood operators with different iteration limits
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Algorithm Functions""")
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Experimental Design

    We run a computational experiment to compare the effectiveness of different neighborhood operators:

    - **Neighborhood operators**: API (adjacent swap) vs. PI (any pair swap)
    - **Iteration limits**: 500 and 1,000 non-improving iterations
    - **Test instances**: All instances in `test_instances_20250930/`
    - **Random seed**: Fixed at 0 for reproducibility

    For each combination:
    1. Generate SPT initial solution
    2. Run neighborhood search
    3. Record the best solution found
    """
    )
    return


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
    # All variables are local to this cell (marimo scoping)
    _data_directory = pathlib.Path('test_instances_20250930/')
    assert _data_directory.exists()

    _possible_neighborhood_functions = {
        'API': compute_API_neighbor,
        'PI': compute_PI_neighbor,
    }

    experiment_results = []

    _data_filepaths = sorted(list(_data_directory.glob('*.csv')))
    for _data_filepath in tqdm(_data_filepaths):
        for _neighborhood_function_name, _neighborhood_function in _possible_neighborhood_functions.items():
            for _max_ni_iterations in [500, 1_000]:

                random.seed(0)

                _data = pl.read_csv(_data_filepath)
                _data_pd = _data.to_pandas().set_index('j')
                _data_dict = _data_pd.to_dict(orient='index')

                _spt_solution = get_SPT_solution(_data)
                _spt_solution_value = compute_weighted_tardiness(
                    data_dict=_data_dict,
                    solution=_spt_solution,
                )

                _best_neighbor_solution = run_neighborhood_search(
                    neighborhood_function=_neighborhood_function,
                    solution=_spt_solution,
                    objective_function=compute_weighted_tardiness,
                    data_dict=_data_dict,
                    max_non_improving_iterations=_max_ni_iterations,
                )
                _best_neighbor_solution_value = compute_weighted_tardiness(
                    data_dict=_data_dict,
                    solution=_best_neighbor_solution,
                )

                experiment_results.append({
                    'filename': _data_filepath.stem,
                    'neighborhood': _neighborhood_function_name,
                    'ni_iterations': _max_ni_iterations,
                    'best_value': _best_neighbor_solution_value,
                })
    return (experiment_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Experimental Results

    The analysis below compares:
    1. **Neighborhood operator effectiveness**: Which operator (API vs PI) finds better solutions?
    2. **Iteration limit impact**: Does running longer (1000 vs 500 iterations) improve results significantly?
    """
    )
    return


@app.cell
def _(experiment_results, pl):
    experiment_results_df = pl.DataFrame(experiment_results)
    return (experiment_results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Comparison by Neighborhood Operator""")
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Comparison by Non-Improving Iteration Limit""")
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
