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
def _(haversine_distances, pathlib, pl):
    data_filepath = pathlib.Path('data/tsp_AL_100.csv')

    data = pl.read_csv(
        data_filepath
    ).with_columns(
        lat_radians = pl.col('lat').radians(),
        lng_radians = pl.col('lng').radians()
    )

    X = data.select([
        'lat_radians',
        'lng_radians',
    ]).to_numpy()

    haverine_distance_array = 3963.1 * haversine_distances(X=X, Y=X)

    distance_df = pl.DataFrame(
        haverine_distance_array,
        schema=data['city'].to_list(),
    ).with_columns(
        pl.Series(name='origin', values=data['city'].to_list())
    ).unpivot(
        index='origin',
        variable_name='destination',
        value_name='distance'
    )
    return (distance_df,)


@app.cell
def _(distance_df, pl):
    all_locations = set(distance_df['origin'].unique().to_list())
    current_location = 'Tuscaloosa'

    visited_locations = [current_location]
    total_distance_traveled = 0
    unvisited_locations = all_locations - set(visited_locations)

    while unvisited_locations:
        _sorted_destinations = distance_df.filter(
            pl.col('origin') == current_location
        ).filter(
            pl.col('destination').is_in(unvisited_locations)
        ).sort(
            'distance'
        )

        next_location = _sorted_destinations.item(row=0, column='destination')
        travel_distance = _sorted_destinations.item(row=0, column='distance')

        total_distance_traveled += travel_distance
        visited_locations.append(next_location)
        unvisited_locations = all_locations - set(visited_locations)

        current_location = next_location

    first_location = visited_locations[0]
    last_location = visited_locations[-1]

    return_distance = distance_df.filter(
        pl.col('origin') == last_location
    ).filter(
        pl.col('destination') == first_location
    ).item(row=0, column='distance')

    total_distance_traveled += return_distance

    print(f' - {total_distance_traveled = :,}')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
