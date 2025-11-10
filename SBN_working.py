import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import pathlib

    import polars as pl

    import sbn_utilities
    return pl, sbn_utilities


@app.cell
def _(pl, sbn_utilities):
    raw_data = pl.read_csv(
        'data/SBN_data.csv'
    ).with_columns(
        pl.col('machine_sequence').str.split(','),
        pl.col('pij').str.split(',')
    )

    _cpm_data = []
    for _row in raw_data.to_dicts():
        _current_job = _row.get('job')
        _previous_machine_job = None
        _row_machine_sequence = _row.get('machine_sequence')
        _row_pij = _row.get('pij')
        for _machine, _pij in zip(_row_machine_sequence, _row_pij):
            _current_machine_job = f'{_machine},{_current_job}'
            if _previous_machine_job is None:
                _predecessors = None
            else:
                _predecessors = [_previous_machine_job]
            _cpm_data.append({
                'current_machine_job': _current_machine_job,
                'predecessors': _predecessors,
                'pij': _pij,
            })
            _previous_machine_job = _current_machine_job

    _cpm_data = pl.DataFrame(
        _cpm_data
    ).with_columns(
        pl.col('pij').cast(pl.Int64)
    )

    cpm_results = sbn_utilities.calculate_cpm(_cpm_data)
    return (cpm_results,)


@app.cell
def _(cpm_results, pl):
    _current_machine = '1'
    _relevant_jobs = []
    for _job in cpm_results.keys():
        if _job.startswith(_current_machine):
            _relevant_jobs.append(_job)

    _current_single_machine_data = []
    for _job in _relevant_jobs:
        _es = cpm_results[_job].get('early_start')
        _ef = cpm_results[_job].get('early_finish')
        _pj = _ef - _es
        _lf = cpm_results[_job].get('late_finish')
        _current_single_machine_data.append({
            'job': _job,
            'pj': _pj,
            'rj': _es,
            'dj': _lf,
        })

    pl.DataFrame(_current_single_machine_data)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
