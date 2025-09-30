import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib
    import random

    import numpy as np
    import polars as pl
    return np, pathlib, pl, random


@app.cell
def _(np, pathlib, pl, random):
    data_directory = pathlib.Path('test_instances_20250930')
    data_directory.mkdir(exist_ok=True)

    num_instances = 25
    for instance in range(1, num_instances+1):
        seed = random.randint(a=0, b=100)
        n = random.randint(a=30, b=150)
    
        np.random.seed(seed)
    
        pj_low = 5
        pj_high = 15
    
        pj_array = np.random.randint(
            low=pj_low,
            high=pj_high,
            size=n
        )
    
        rjs = []
        for i in range(n):
            if i == 0:
                rjs.append(np.random.randint(0, (pj_low + pj_high)/2))
            else:
                rjs.append(rjs[i-1] + np.random.randint(0, (pj_low + pj_high)/2))
        rj_array = np.array(rjs)
        np.random.shuffle(rj_array)
    
        dj_offset = np.random.randint(
            low=0,
            high=pj_high,
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
        data_filepath = pathlib.Path(
            data_directory,
            f'instance-{n}-{seed}.csv'
        )
        data.write_csv(data_filepath)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
