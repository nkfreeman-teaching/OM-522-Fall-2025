import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import pathlib

    import pandas as pd
    import polars as pl

    import sbn_utilities
    return pd, pl, sbn_utilities


@app.cell
def _(pd, pl, sbn_utilities):
    raw_data = pl.read_csv(
        'data/SBN_data.csv'
    ).with_columns(
        pl.col('machine_sequence').str.split(','),
        pl.col('pij').str.split(',')
    )

    machine_list = raw_data['machine_sequence'].explode().unique().sort().to_list()
    scheduled_machines = []

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

    while set(machine_list) - set(scheduled_machines):
        unscheduled_machines = list(set(machine_list) - set(scheduled_machines))

        cpm_results = sbn_utilities.calculate_cpm(_cpm_data)
    
        _all_single_machine_results = []
        for _current_machine in unscheduled_machines:
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
    
            _current_single_machine_data = pl.DataFrame(_current_single_machine_data)
            _single_machine_results = sbn_utilities.minimize_maximum_lateness(_current_single_machine_data)
            _sequence = _single_machine_results.get('sequence')
            _lmax = _single_machine_results.get('max_lateness')
    
            _all_single_machine_results.append({
                'machine': _current_machine,
                'sequence': _sequence,
                'Lmax': _lmax,
            })
    
        _all_single_machine_results = pl.DataFrame(
            _all_single_machine_results
        ).sort(
            'Lmax', 
            descending=True
        )
        _bottleneck_information = _all_single_machine_results.to_dicts()[0]
        _bottleneck_machine = _bottleneck_information.get('machine')
        _bottleneck_sequence = _bottleneck_information.get('sequence')
        print(f'Identified bottleneck is {_bottleneck_machine}')
    
        _cpm_data_dicts = _cpm_data.to_pandas().set_index(
            'current_machine_job'
        ).to_dict(
            orient='index'
        )
        for key in _cpm_data_dicts:
            if _cpm_data_dicts[key]['predecessors'] is None:
                pass
            else:
                _cpm_data_dicts[key]['predecessors'] = _cpm_data_dicts[key]['predecessors'].tolist()
        for _new_predecessor, _target in zip(_bottleneck_sequence[:-1], _bottleneck_sequence[1:]):
            if _cpm_data_dicts[_target]['predecessors']:
                _cpm_data_dicts[_target]['predecessors'].append(_new_predecessor)
            else:
                _cpm_data_dicts[_target]['predecessors'] = [_new_predecessor]
    
        _cpm_data = pd.DataFrame.from_dict(
            _cpm_data_dicts, 
            orient='index'
        ).reset_index()
    
        _cpm_data = pl.from_pandas(
            _cpm_data
        ).rename({
            'index': 'current_machine_job'
        })

        print(_cpm_data)

        scheduled_machines.append(_bottleneck_machine)
    return


@app.cell
def _(a):
    a
    return


if __name__ == "__main__":
    app.run()
