import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pathlib

    import polars as pl
    return mo, pathlib, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # TSP Instance Generator

    This utility generates TSP instances from US city data.

    ## Data Source

    **Note**: This script requires the SimpleMaps US Cities database, which is not included in the repository.

    Download from: https://simplemaps.com/data/us-cities

    Place the CSV file at the path specified in the code below, or modify the path to match your local setup.

    ## Parameters

    - **State**: AL (Alabama)
    - **Number of cities (n)**: 100 (top cities by population)

    ## Output

    Creates `data/tsp_{state}_{n}.csv` with columns: city, state_id, county_name, county_fips, population, lat, lng
    """
    )
    return


@app.cell
def _(pathlib, pl):
    _state = 'AL'
    _n = 100

    # NOTE: Update this path to point to your local copy of the SimpleMaps US Cities database
    _data = pl.read_csv(
        '/home/nick/Dropbox/Data/simplemaps_uscities_prov1.6/uscities.csv',
        infer_schema_length=None,
        null_values=''
    ).filter(
        pl.col('state_id') == _state
    ).drop_nulls(
        'population'
    ).sort(
        'population',
        descending=True,
    ).unique(
        'city',
        keep='first',
        maintain_order=True
    ).head(
        n=_n
    ).select([
        'city',
        'state_id',
        'county_name',
        'county_fips',
        'population',
        'lat',
        'lng'
    ])

    _data_filepath = pathlib.Path(f'data/tsp_{_state}_{_n}.csv')
    if not _data_filepath.exists():
        _data.write_csv(_data_filepath)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
