import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib

    import polars as pl
    from sklearn.metrics.pairwise import haversine_distances
    return haversine_distances, pathlib, pl


@app.cell
def _(pathlib):
    data_filepath = pathlib.Path('data/tsp_AL_100.csv')
    data_filepath.exists()
    return (data_filepath,)


@app.cell
def _(data_filepath, pl):
    data = pl.read_csv(
        data_filepath
    ).with_columns(
        lat_radians = pl.col('lat').radians(),
        lng_radians = pl.col('lng').radians()
    )
    return (data,)


@app.cell
def _(data, haversine_distances):
    X = data.select([
        'lat_radians',
        'lng_radians',
    ]).to_numpy()

    haverine_distance_array = 3963.1 * haversine_distances(X=X, Y=X)
    return (haverine_distance_array,)


@app.cell
def _(data, haverine_distance_array, pl):
    pl.DataFrame(
        haverine_distance_array,
        schema=data['city'].to_list(),
    ).with_columns(
        pl.Series(name='origin', values=data['city'].to_list())
    ).unpivot(
        index='origin',
        variable_name='destination',
        value_name='distance'
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
