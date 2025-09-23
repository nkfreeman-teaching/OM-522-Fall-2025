import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import pathlib
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from sklearn.metrics.pairwise import haversine_distances

    sns.set_style('whitegrid')
    return haversine_distances, mo, np, pathlib, pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Function Definitions""")
    return


@app.cell
def _(haversine_distances, np, pl, plt):
    def visualize_tsp_solution(
        tour_list: list,
        coordinate_df: pl.DataFrame,
        figsize: tuple[int, int] = (12, 8)
    ):
        """
        Alternative visualization with different styling.
        """

        df = get_solution_df(
            tour_list=tour_list,
            coordinate_df=coordinate_df,
        )

        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        # Extract coordinates
        lngs = df['lng'].to_list()
        lats = df['lat'].to_list()
        cities = df['city'].to_list()

        n_cities = len(cities)

        # Draw the tour path first (so it appears behind points)
        for i in range(n_cities):
            start_lng, start_lat = lngs[i], lats[i]
            end_lng, end_lat = lngs[(i + 1) % n_cities], lats[(i + 1) % n_cities]

            # Draw line segment with gradient color
            ax.plot(
                [start_lng, end_lng], 
                [start_lat, end_lat], 
                color='k', 
                linewidth=2, 
                alpha=0.7,
            )

            # Add arrow in the middle of each segment
            mid_lng = (start_lng + end_lng) / 2
            mid_lat = (start_lat + end_lat) / 2
            dx = end_lng - start_lng
            dy = end_lat - start_lat

            # Normalize arrow direction
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = dx / length * 0.1  # Scale arrow size
                dy_norm = dy / length * 0.1

                ax.annotate(
                    '', 
                    xy=(mid_lng + dx_norm, mid_lat + dy_norm),
                    xytext=(mid_lng - dx_norm, mid_lat - dy_norm),
                    arrowprops=dict(arrowstyle='->', color='k', lw=2)
                )

        # Create the scatterplot for cities with numbers
        for i, (lng, lat, city) in enumerate(zip(lngs, lats, cities)):
            ax.scatter(
                lng, 
                lat, 
                c='white', 
                s=50, 
                zorder=5, 
                edgecolors='black', 
                linewidth=2,
            )

        # Customize the plot
        ax.set_xlabel(
            'Longitude', 
            fontsize=12,
        )
        ax.set_ylabel(
            'Latitude', 
            fontsize=12,
        )
        ax.set_title(
            'TSP Solution', 
            fontsize=14, 
            fontweight='bold',
        )
        ax.grid(
            True, 
            alpha=0.3,
        )

        plt.tight_layout()
        return fig, ax


    def get_distance_df(
        coordinate_df: pl.DataFrame,
    ) -> pl.DataFrame:

        data = coordinate_df.clone().with_columns(
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

        return distance_df


    def get_distance_dict(
        distance_df: pl.DataFrame,
    ) -> dict:

        distance_dict = {}
        for _entry in distance_df.to_dicts():
            _origin = _entry.get('origin')
            _destination = _entry.get('destination')
            _distance = _entry.get('distance')
            distance_dict[(_origin, _destination)] = _distance

        return distance_dict


    def get_nearest_neighbors_solution(
        distance_df: pl.DataFrame,
        start_location: str,
    ) -> dict:

        all_locations = set(distance_df['origin'].unique().to_list())

        current_location = start_location
        visited_locations = [current_location]
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

            visited_locations.append(next_location)
            unvisited_locations = all_locations - set(visited_locations)

            current_location = next_location

        first_location = visited_locations[0]
        last_location = visited_locations[-1]

        return visited_locations


    def get_solution_df(
        tour_list: list,
        coordinate_df: pl.DataFrame,
    ):
        _city_info = coordinate_df.select(
            ['city', 'lat', 'lng']
        ).to_dicts()

        _city_lat_lng_dict = {}
        for _item in _city_info:
            _item_city = _item.get('city')
            _item_lat = _item.get('lat')
            _item_lng = _item.get('lng')

            _city_lat_lng_dict[_item_city] = {
                'lat': _item_lat,
                'lng': _item_lng,
            }

        solution_df = []
        for _city in tour_list:
            if _city_lat_lng_dict.get(_city):
                solution_df.append({
                    'city': _city,
                    'lng': _city_lat_lng_dict.get(_city).get('lng'),
                    'lat': _city_lat_lng_dict.get(_city).get('lat'),
                })

        solution_df = pl.DataFrame(solution_df)

        return solution_df


    def compute_tour_distance(
        distance_dict: dict,
        tour_list: list,
    ) -> float:

        distance = 0
        for pair in zip(tour_list[:-1], tour_list[1:]):
            distance += distance_dict.get(pair)
        distance += distance_dict.get((tour_list[-1], tour_list[0]))

        return distance
    return (
        compute_tour_distance,
        get_distance_df,
        get_distance_dict,
        get_nearest_neighbors_solution,
        visualize_tsp_solution,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data Setup""")
    return


@app.cell
def _(get_distance_df, get_distance_dict, pathlib, pl):
    data_filepath = pathlib.Path('data/tsp_AL_100.csv')
    coordinate_df = pl.read_csv(data_filepath)

    distance_df = get_distance_df(
        coordinate_df=coordinate_df
    )

    distance_dict = get_distance_dict(
        distance_df=distance_df
    )
    return coordinate_df, distance_df, distance_dict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Construction Heuristic - Nearest Neighbors""")
    return


@app.cell
def _(coordinate_df, mo):
    city_selection = mo.ui.dropdown(
        options=coordinate_df['city'].unique().sort().to_list(),
        value='Tuscaloosa',
    )
    city_selection
    return (city_selection,)


@app.cell
def _(
    city_selection,
    compute_tour_distance,
    coordinate_df,
    distance_df,
    distance_dict,
    get_nearest_neighbors_solution,
    visualize_tsp_solution,
):
    nn_solution = get_nearest_neighbors_solution(
        distance_df=distance_df,
        start_location=city_selection.value,
    )
    nn_distance = compute_tour_distance(
        distance_dict=distance_dict,
        tour_list=nn_solution,
    )
    print(f' - Created tour with distance {nn_distance:.2f}')

    visualize_tsp_solution(
        tour_list=nn_solution,
        coordinate_df=coordinate_df,
        figsize=(4.5, 6)
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
