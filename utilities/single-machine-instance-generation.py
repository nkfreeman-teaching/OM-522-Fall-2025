import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pathlib
    import random

    import numpy as np
    import polars as pl
    return mo, np, pathlib, pl, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Single Machine Instance Generator

    This utility generates random test instances for single machine scheduling problems.

    ## Instance Parameters

    - **Number of instances**: 25
    - **Jobs per instance (n)**: Random between 30-150
    - **Processing times (pⱼ)**: Random between 5-15
    - **Release times (rⱼ)**: Sequentially generated with random offsets
    - **Due dates (dⱼ)**: rⱼ + pⱼ + random offset (0 to 15)
    - **Weights (wⱼ)**: Random between 1-10

    ## Output

    Instances are saved to `test_instances_20250930/` as CSV files with naming convention:
    `instance-{n}-{seed}.csv`
    """
    )
    return


@app.cell
def _(np, pathlib, pl, random):
    _data_directory = pathlib.Path('test_instances_20250930')
    _data_directory.mkdir(exist_ok=True)

    _num_instances = 25
    for _instance in range(1, _num_instances+1):
        _seed = random.randint(a=0, b=100)
        _n = random.randint(a=30, b=150)

        np.random.seed(_seed)

        _pj_low = 5
        _pj_high = 15

        _pj_array = np.random.randint(
            low=_pj_low,
            high=_pj_high,
            size=_n
        )

        _rjs = []
        for _i in range(_n):
            if _i == 0:
                _rjs.append(np.random.randint(0, (_pj_low + _pj_high)/2))
            else:
                _rjs.append(_rjs[_i-1] + np.random.randint(0, (_pj_low + _pj_high)/2))
        _rj_array = np.array(_rjs)
        np.random.shuffle(_rj_array)

        _dj_offset = np.random.randint(
            low=0,
            high=_pj_high,
            size=_n
        )

        _dj_array = _rj_array + _pj_array + _dj_offset

        _wj_array = np.random.randint(
            low=1,
            high=10,
            size=_n
        )

        _data = pl.DataFrame({
            'j': [_i for _i in range(1, _n + 1)],
            'pj': _pj_array,
            'rj': _rj_array,
            'dj': _dj_array,
            'wj': _wj_array,
        })
        _data_filepath = pathlib.Path(
            _data_directory,
            f'instance-{_n}-{_seed}.csv'
        )
        _data.write_csv(_data_filepath)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
