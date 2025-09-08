import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    from sklearn.metrics.pairwise import haversine_distances

    sns.set_style('whitegrid')
    return haversine_distances, pl, plt, sns


@app.cell
def _(pl, plt, sns):
    data = pl.read_csv(
        'data/tsp_AL_100.csv'
    ).with_columns(
        lat_radians = pl.col('lat').radians(),
        lng_radians = pl.col('lng').radians(),
    ).with_columns(
        city_state = pl.col('city') + ', ' + pl.col('state_id')
    )

    _fig, _ax = plt.subplots(1, 1, figsize=(4, 6))

    sns.scatterplot(
        data,
        x='lng',
        y='lat',
        size=data['population'],
        sizes=(50, 250),
        alpha=0.5,
        edgecolor='k',
        color='steelblue',
        legend=False,
    )
    return (data,)


@app.cell
def _(data, haversine_distances, pl):
    lat_lng_array = data.select([
        'lat_radians',
        'lng_radians',
    ]).to_numpy()

    distance_matrix = haversine_distances(
        lat_lng_array
    )
    distance_matrix = distance_matrix * (3963.1)

    distance_df = pl.DataFrame(
        distance_matrix,
        schema=data['city_state'].to_list()
    ).with_columns(
        origin = pl.Series(name='origin', values=data['city_state'].to_list())
    ).unpivot(
        index='origin',
        variable_name='destination',
        value_name='distance'
    )


    target_city = 'Tuscaloosa, AL'

    closest_city = distance_df.filter(
        pl.col('origin') == target_city,
        pl.col('destination') != target_city
    ).sort(
        'distance'
    ).item(0, 'destination')

    print(f'The closest city to {target_city} is {closest_city}')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
