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
    from tqdm.auto import tqdm

    sns.set_style('whitegrid')
    return haversine_distances, mo, np, pathlib, pl, plt, random, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Traveling Salesman Problem (TSP) with Neighborhood Search

    This notebook solves the **Traveling Salesman Problem (TSP)** using construction heuristics and neighborhood search.

    ## Problem Description

    Given a set of cities with coordinates (latitude, longitude):
    - **Objective**: Find the shortest tour that visits each city exactly once and returns to the starting city
    - **Distance metric**: Haversine distance (great-circle distance on Earth's surface)

    ## Solution Approach

    1. **Construction Heuristic**: Nearest Neighbor algorithm
       - Start from a given city
       - Repeatedly visit the nearest unvisited city
       - Return to the starting city

    2. **Improvement**: Neighborhood search with SSR (String String Reversal)
       - SSR: Reverse a substring of the tour
       - Accept if tour distance improves
       - Try multiple starting cities to find the best solution
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Function Definitions""")
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
                s=10, 
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
    mo.md(
        r"""
    ## Data Setup

    We load geographic data for 100 cities in Alabama (AL). The dataset includes:
    - **city**: City name
    - **lat**: Latitude coordinate
    - **lng**: Longitude coordinate

    We precompute all pairwise distances using the Haversine formula (distance on Earth's surface in miles).
    """
    )
    return


@app.cell
def _(get_distance_df, get_distance_dict, pathlib, pl):
    _data_filepath = pathlib.Path('data/tsp_AL_100.csv')
    coordinate_df = pl.read_csv(_data_filepath)

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
def _(
    compute_tour_distance,
    coordinate_df,
    distance_df,
    distance_dict,
    get_nearest_neighbors_solution,
    visualize_tsp_solution,
):
    nn_solution = get_nearest_neighbors_solution(
        distance_df=distance_df,
        start_location='Tuscaloosa'
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Neighborhood Search

    ## Functions to Generate Neighbors
    """
    )
    return


@app.cell
def _(random):
    def generate_API_neighbor(incumbent_solution):

        random_index = random.randint(0, len(incumbent_solution) - 2)

        neighbor = list(incumbent_solution)

        neighbor[random_index], neighbor[random_index+1] = neighbor[random_index+1], neighbor[random_index]

        return neighbor


    def generate_PI_neighbor(incumbent_solution):

        index1 = random.randint(0, len(incumbent_solution) - 1)
        index2 = random.randint(0, len(incumbent_solution) - 1)

        neighbor = list(incumbent_solution)

        neighbor[index1], neighbor[index2] = neighbor[index2], neighbor[index1]

        return neighbor


    def generate_SSR_neighbor(incumbent_solution):

        index1 = random.randint(0, len(incumbent_solution) - 1)
        index2 = random.randint(0, len(incumbent_solution) - 1)
        if index1 > index2:
            index1, index2 = index2, index1

        neighbor = list(incumbent_solution)

        neighbor[index1:index2+1] = neighbor[index1:index2+1][::-1]

        return neighbor
    return (generate_SSR_neighbor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Function to Conduct Neighborhood Search""")
    return


@app.cell
def _(compute_tour_distance, distance_dict):
    def run_neighborhood_search(
        initial_solution: list,
        max_non_improving_iterations: int,
        neighborhood_function,
    ) -> dict:

        incumbent_solution = list(initial_solution)
        incumbent_value = compute_tour_distance(
            distance_dict=distance_dict,
            tour_list=incumbent_solution,
        )
        ni_iterations = 0
        while ni_iterations < max_non_improving_iterations:
            ni_iterations += 1

            neighbor = neighborhood_function(incumbent_solution)
            neighbor_value = compute_tour_distance(
                distance_dict=distance_dict,
                tour_list=neighbor,
            )
            if neighbor_value < incumbent_value:
                incumbent_solution = list(neighbor)
                incumbent_value = neighbor_value
                ni_iterations = 0

        return {
            'incumbent': incumbent_solution,
            'incumbent_value': incumbent_value
        }
    return (run_neighborhood_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Multi-Start Neighborhood Search

    To find a high-quality solution, we:
    1. Try nearest neighbor construction from **every city** as a starting point
    2. Improve each construction with SSR neighborhood search (5,000 non-improving iterations)
    3. Track the best solution found across all starting points

    This multi-start approach helps avoid poor local optima that may result from a single starting location.
    """
    )
    return


@app.cell
def _(
    coordinate_df,
    distance_df,
    generate_SSR_neighbor,
    get_nearest_neighbors_solution,
    random,
    run_neighborhood_search,
    tqdm,
    visualize_tsp_solution,
):
    # All variables are local to this cell (marimo scoping)
    random.seed(42)

    _possible_starting_locations = coordinate_df['city'].to_list()

    _incumbent = None
    _incumbent_value = None
    for _possible_starting_location in tqdm(_possible_starting_locations):
        _nearest_neighbor_solution = get_nearest_neighbors_solution(
            distance_df=distance_df,
            start_location=_possible_starting_location,
        )
        _neighborhood_search_results = run_neighborhood_search(
            initial_solution = _nearest_neighbor_solution,
            max_non_improving_iterations=5_000,
            neighborhood_function=generate_SSR_neighbor,
        )
        _best_neighborhood_search_solution = _neighborhood_search_results.get('incumbent')
        _best_neighborhood_search_value = _neighborhood_search_results.get('incumbent_value')

        # Update incumbent if this is the first solution or if we found a better (shorter) tour
        if (_incumbent_value is None) or (_incumbent_value > _best_neighborhood_search_value):
            _incumbent_value = _best_neighborhood_search_value
            _incumbent = list(_best_neighborhood_search_solution)

    print(f'Best tour distance found: {_incumbent_value:.2f} miles')

    visualize_tsp_solution(
        tour_list=_incumbent,
        coordinate_df=coordinate_df,
        figsize=(4.5, 6)
    )
    return


if __name__ == "__main__":
    app.run()
