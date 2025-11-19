import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import itertools
    import pathlib
    import random

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from sklearn.metrics.pairwise import haversine_distances
    from tqdm.auto import tqdm

    sns.set_style('whitegrid')
    return haversine_distances, itertools, mo, np, pathlib, pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Clarke-Wright Savings Algorithm for Vehicle Routing

    This notebook solves the **Capacitated Vehicle Routing Problem (CVRP)** using the Clarke-Wright Savings algorithm.

    ## Problem Description

    Given:
    - A **depot** (central location where all vehicles start and end)
    - A set of **customers** with geographic coordinates
    - **Vehicle capacity** constraint (maximum customers per route)

    **Objective**: Create routes that:
    - Visit all customers exactly once
    - Start and end at the depot
    - Respect vehicle capacity limits
    - Minimize total travel distance

    ## Solution Approach

    1. **Initialize**: Each customer starts in their own route (depot → customer → depot)

    2. **Calculate Savings**: For each pair of customers (i, j), compute:
       - Savings s(i,j) = d(depot, i) + d(j, depot) - d(i, j)
       - This represents the distance saved by merging routes

    3. **Merge Routes**: Process savings in descending order:
       - If merging two routes doesn't exceed capacity, combine them
       - Update cluster assignments

    4. **Optimize Routes**: Use Nearest Neighbor heuristic within each cluster to sequence visits
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Utility Functions

    This section defines helper functions for:
    - **Distance calculation**: Haversine formula for geographic distances
    - **Route visualization**: Plotting routes with color-coded clusters
    - **Solution construction**: Nearest Neighbor heuristic for sequencing
    """
    )
    return


@app.cell
def _(haversine_distances, np, pl, plt):
    def visualize_solution(
        cluster_routes: dict,
        coordinate_df: pl.DataFrame,
        figsize: tuple[int, int] = (12, 8)
    ):
        """
        Alternative visualization with different styling.
        """

        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        # Get a colormap object
        cmap = plt.get_cmap('viridis')

        # Get the color for the first element in a sequence
        # np.linspace generates evenly spaced values in the [0, 1] range
        num_colors = len(cluster_routes)
        colors_list = [cmap(i) for i in np.linspace(0, 1, num_colors)]

        for _cluster, _cluster_tour in cluster_routes.items():

            df = get_solution_df(
                tour_list=_cluster_tour,
                coordinate_df=coordinate_df,
            )

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
                    color=colors_list[_cluster-1], 
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
            'CW Savings Solution', 
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
        get_distance_df,
        get_distance_dict,
        get_nearest_neighbors_solution,
        get_solution_df,
        visualize_solution,
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
    mo.md(
        r"""
    ## Problem Setup

    We define:
    - **Depot**: Birmingham (central hub where all routes start and end)
    - **Customers**: All other cities that need to be visited
    """
    )
    return


@app.cell
def _(coordinate_df, pl):
    depot = 'Birmingham'

    customer_locations = coordinate_df.filter(
        pl.col('city') != depot,
    ).get_column(
        'city'
    )

    print(f'Depot: {depot}')
    print(f' - {len(customer_locations)} customers')
    return customer_locations, depot


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 1: Calculate Savings

    For each pair of customers (i, j), we calculate the **savings** from merging their routes:

    **s(i, j) = d(depot, i) + d(j, depot) - d(i, j)**

    This represents the distance saved by visiting customers i and j in sequence instead of making separate trips from the depot.

    The savings are sorted in descending order to prioritize the most beneficial merges.
    """
    )
    return


@app.cell
def _(customer_locations, depot, distance_dict, itertools, pl):
    savings_data = []
    for customer1, customer2 in itertools.combinations(customer_locations, r=2):
        savings_data.append({
            'customer1': customer1,
            'customer2': customer2,
            'savings': (
                distance_dict[(depot, customer1)] 
                + distance_dict[(customer2, depot)] 
                - distance_dict[(customer1, customer2)]
            ),
        })

    savings_data = pl.DataFrame(
        savings_data
    ).sort(
        'savings',
        descending=True
    )

    savings_data.head()
    return (savings_data,)


@app.function
def get_customer2cluster_mapping(
    cluster2customers: dict,
) -> dict:

    _customer2cluster = {}
    for _idx, _customer_list in cluster2customers.items():
        if _customer_list:
            for _customer in _customer_list:
                _customer2cluster[_customer] = _idx

    return _customer2cluster


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 2: Merge Routes (Clarke-Wright Algorithm)

    Starting with each customer in their own route, we iteratively merge routes:

    1. **Initialize**: Each customer forms a single-customer cluster
    2. **Process savings** in descending order:
       - For each pair (i, j) with positive savings
       - Check if merging their clusters respects the **capacity constraint**
       - If yes, merge the two clusters
    3. **Result**: Clusters of customers that will be served by the same vehicle

    **Capacity constraint**: Maximum of 5 customers per route (vehicle capacity)
    """
    )
    return


@app.cell
def _(customer_locations, depot, savings_data):
    CAPACITY = 5

    cluster2customers = {_idx: [_customer] for _idx, _customer in enumerate(customer_locations, 1)}
    customer2cluster = get_customer2cluster_mapping(cluster2customers)

    for _entry in savings_data.to_dicts():
        _customer1 = _entry.get('customer1')
        _customer2 = _entry.get('customer2')

        _customer1_cluster_idx = customer2cluster.get(_customer1)
        _customer1_cluster_members = cluster2customers.get(_customer1_cluster_idx)

        _customer2_cluster_idx = customer2cluster.get(_customer2)
        _customer2_cluster_members = cluster2customers.get(_customer2_cluster_idx)

        if _customer1_cluster_idx != _customer2_cluster_idx:
            if (len(_customer1_cluster_members) + len(_customer2_cluster_members)) <= CAPACITY:
                cluster2customers[_customer1_cluster_idx].extend(_customer2_cluster_members)
                cluster2customers[_customer2_cluster_idx] = None
                customer2cluster = get_customer2cluster_mapping(cluster2customers)


    cluster2customers = {_key: _val for _key, _val in cluster2customers.items() if _val}
    cluster2customers = {_idx: _val for _idx, _val in enumerate(cluster2customers.values(), 1)}
    for _key in cluster2customers:
        cluster2customers[_key].append(depot)
    return (cluster2customers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 3: Generate Routes Within Clusters

    For each cluster, we determine the visiting sequence using the **Nearest Neighbor heuristic**:

    1. Start at the depot
    2. Visit the nearest unvisited customer in the cluster
    3. Repeat until all customers in the cluster are visited
    4. Return to depot

    This creates efficient routes within each cluster while respecting the cluster assignments from the savings algorithm.
    """
    )
    return


@app.cell
def _(
    cluster2customers,
    depot,
    distance_df,
    get_nearest_neighbors_solution,
    pl,
):
    routes = {}
    for _cluster, _customer_list in cluster2customers.items():

        cluster_distance_df = distance_df.filter(
            pl.col('origin').is_in(_customer_list),
            pl.col('destination').is_in(_customer_list)
        )

        routes[_cluster] = get_nearest_neighbors_solution(
            distance_df=cluster_distance_df, 
            start_location=depot,
        )
    return (routes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Results

    The visualization below shows the final solution:
    - Each **color** represents a different vehicle route (cluster)
    - All routes start and end at the **depot** (Birmingham)
    - **Arrows** indicate the direction of travel
    - The algorithm has grouped nearby customers together while respecting capacity constraints
    """
    )
    return


@app.cell
def _(coordinate_df, get_solution_df, routes):
    get_solution_df(coordinate_df=coordinate_df, tour_list=routes[1])
    return


@app.cell
def _(coordinate_df, routes, visualize_solution):
    visualize_solution(
        cluster_routes=routes, 
        coordinate_df=coordinate_df,
        figsize=(5, 8)
    )
    return


if __name__ == "__main__":
    app.run()
