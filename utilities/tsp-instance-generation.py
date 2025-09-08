import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib

    import polars as pl
    return pathlib, pl


@app.cell
def _(pathlib, pl):
    state = 'AL'
    n = 100

    data = pl.read_csv(
        '/home/nick/Dropbox/Data/simplemaps_uscities_prov1.6/uscities.csv',
        infer_schema_length=None,
        null_values=''
    ).filter(
        pl.col('state_id') == state
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
        n=n
    ).select([
        'city',
        'state_id',
        'county_name',
        'county_fips',
        'population',
        'lat',
        'lng'
    ])

    data_filepath = pathlib.Path(f'data/tsp_{state}_{n}.csv')
    if not data_filepath.exists():
        data.write_csv(data_filepath)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
