import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Critical Path Method (CPM) Analysis

    This notebook implements the Critical Path Method for project scheduling.
    The CPM algorithm helps identify which tasks are critical to completing
    a project on time and calculates the earliest and latest start/finish
    times for each task.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports

    We use the following packages:
    - **polars**: For efficient data loading and manipulation
    - **matplotlib**: For creating the Gantt-style schedule visualization
    - **collections**: For data structures (defaultdict, deque)
    """)
    return


@app.cell
def _():
    import polars as pl
    import matplotlib.pyplot as plt
    from collections import defaultdict, deque
    import marimo as mo
    return defaultdict, mo, pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function Definitions

    This section contains all the functions that implement the CPM algorithm:

    1. **load_project_data**: Loads the CSV file with project data
    2. **parse_predecessors**: Parses predecessor dependencies from strings
    3. **calculate_early_times**: Forward pass - calculates ES and EF times
    4. **calculate_late_times**: Backward pass - calculates LS and LF times
    5. **calculate_slack**: Computes slack time (float) for each task
    6. **find_critical_path**: Identifies tasks with zero slack
    7. **print_results**: Displays formatted results table
    8. **visualize_schedule**: Creates a Gantt chart visualization
    """)
    return


@app.cell
def _(defaultdict, pl, plt):
    def load_project_data(filename):
        """
        Load project data from CSV file.

        Returns:
            polars.DataFrame: DataFrame with columns j (job), pj (duration), pred (predecessors)
        """
        df = pl.read_csv(filename)
        return df


    def parse_predecessors(pred_str):
        """
        Parse the predecessor string into a list of integers.

        Args:
            pred_str: String containing comma-separated predecessor job numbers

        Returns:
            list: List of predecessor job numbers as integers
        """
        if pred_str is None or pred_str == "":
            return []
        return [int(p.strip()) for p in str(pred_str).split(",")]


    def calculate_early_times(df):
        """
        Calculate Early Start (ES) and Early Finish (EF) times for all tasks.

        The early start is the earliest a task can begin (after all predecessors finish).
        The early finish is early start + duration.

        Args:
            df: DataFrame with project data

        Returns:
            dict: Dictionary mapping job number to (ES, EF) tuple
        """
        # Dictionary to store ES and EF for each job
        early_times = {}

        # Process jobs in order (assumes jobs are numbered sequentially)
        for row in df.iter_rows(named=True):
            job = row['j']
            duration = row['pj']
            predecessors = parse_predecessors(row['pred'])

            # Calculate Early Start (ES)
            if not predecessors:
                # No predecessors means this job can start at time 0
                es = 0
            else:
                # ES is the maximum EF of all predecessors
                es = max(early_times[pred][1] for pred in predecessors)

            # Calculate Early Finish (EF)
            ef = es + duration

            early_times[job] = (es, ef)

        return early_times


    def calculate_late_times(df, early_times, project_duration):
        """
        Calculate Late Start (LS) and Late Finish (LF) times for all tasks.

        The late finish is the latest a task can finish without delaying the project.
        The late start is late finish - duration.

        Args:
            df: DataFrame with project data
            early_times: Dictionary of early start/finish times
            project_duration: Total project duration (maximum EF)

        Returns:
            dict: Dictionary mapping job number to (LS, LF) tuple
        """
        # Dictionary to store LS and LF for each job
        late_times = {}

        # Create a mapping of which jobs depend on each job (reverse dependencies)
        successors = defaultdict(list)
        for row in df.iter_rows(named=True):
            job = row['j']
            predecessors = parse_predecessors(row['pred'])
            for pred in predecessors:
                successors[pred].append(job)

        # Process jobs in reverse order
        jobs = df['j'].to_list()
        for job in reversed(jobs):
            duration = df.filter(pl.col('j') == job)['pj'][0]

            # Calculate Late Finish (LF)
            if job not in successors or not successors[job]:
                # No successors means this is a final job
                lf = project_duration
            else:
                # LF is the minimum LS of all successors
                lf = min(late_times[succ][0] for succ in successors[job])

            # Calculate Late Start (LS)
            ls = lf - duration

            late_times[job] = (ls, lf)

        return late_times


    def calculate_slack(early_times, late_times):
        """
        Calculate slack (float) for each task.

        Slack = LS - ES = LF - EF
        Tasks with zero slack are on the critical path.

        Args:
            early_times: Dictionary of early start/finish times
            late_times: Dictionary of late start/finish times

        Returns:
            dict: Dictionary mapping job number to slack time
        """
        slack = {}
        for job in early_times:
            es, ef = early_times[job]
            ls, lf = late_times[job]
            slack[job] = ls - es  # or equivalently: lf - ef

        return slack


    def find_critical_path(df, slack):
        """
        Identify the critical path (tasks with zero slack).

        Args:
            df: DataFrame with project data
            slack: Dictionary of slack times

        Returns:
            list: List of job numbers on the critical path
        """
        critical_jobs = [job for job, s in slack.items() if s == 0]
        return sorted(critical_jobs)


    def print_results(df, early_times, late_times, slack, critical_path):
        """
        Print the CPM analysis results in a formatted table.
        """
        print("\n" + "="*80)
        print("CRITICAL PATH METHOD (CPM) ANALYSIS")
        print("="*80)
        print()
        print(f"{'Job':<6} {'Duration':<10} {'ES':<8} {'EF':<8} {'LS':<8} {'LF':<8} {'Slack':<8} {'Critical'}")
        print("-"*80)

        for row in df.iter_rows(named=True):
            job = row['j']
            duration = row['pj']
            es, ef = early_times[job]
            ls, lf = late_times[job]
            s = slack[job]
            is_critical = "YES" if s == 0 else ""

            print(f"{job:<6} {duration:<10} {es:<8} {ef:<8} {ls:<8} {lf:<8} {s:<8} {is_critical}")

        print()
        print(f"Project Duration: {max(ef for es, ef in early_times.values())} time units")
        print(f"Critical Path: {' -> '.join(map(str, critical_path))}")
        print("="*80)


    def visualize_schedule(df, early_times, late_times, slack):
        """
        Create a Gantt-style chart showing the project schedule.
        """
        jobs = df['j'].to_list()

        fig, ax = plt.subplots(figsize=(12, len(jobs) * 0.5))

        # Plot each job
        for i, job in enumerate(jobs):
            es, ef = early_times[job]
            ls, lf = late_times[job]
            duration = ef - es
            s = slack[job]

            # Color based on whether job is critical
            color = 'red' if s == 0 else 'lightblue'

            # Draw the early start bar
            ax.barh(i, duration, left=es, height=0.4, color=color,
                    edgecolor='black', label='Critical' if s == 0 and i == 0 else '')

            # Draw slack as a lighter bar if it exists
            if s > 0:
                ax.barh(i, s, left=ef, height=0.4, color='lightgray',
                        edgecolor='gray', alpha=0.5)

        # Formatting
        ax.set_yticks(range(len(jobs)))
        ax.set_yticklabels([f"Job {j}" for j in jobs])
        ax.set_xlabel('Time Units')
        ax.set_title('Project Schedule (CPM)')
        ax.grid(axis='x', alpha=0.3)

        # Add legend outside the plot area
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label='Critical Path'),
            Patch(facecolor='lightblue', edgecolor='black', label='Non-Critical'),
            Patch(facecolor='lightgray', edgecolor='gray', alpha=0.5, label='Slack Time')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        return fig
    return (
        calculate_early_times,
        calculate_late_times,
        calculate_slack,
        find_critical_path,
        load_project_data,
        print_results,
        visualize_schedule,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CPM Analysis Execution

    This section executes the CPM algorithm:

    1. Loads the project data from `project_data.csv`
    2. Performs the forward pass to calculate early times
    3. Performs the backward pass to calculate late times
    4. Calculates slack and identifies the critical path
    5. Displays results and creates the visualization
    """)
    return


@app.cell
def _(
    calculate_early_times,
    calculate_late_times,
    calculate_slack,
    find_critical_path,
    load_project_data,
    mo,
    print_results,
    visualize_schedule,
):
    # Load the project data
    print("Loading project data from 'project_data.csv'...")
    df = load_project_data('project_data.csv')

    # Calculate early times (forward pass)
    print("Calculating early start and finish times...")
    early_times = calculate_early_times(df)

    # Determine project duration
    project_duration = max(ef for es, ef in early_times.values())

    # Calculate late times (backward pass)
    print("Calculating late start and finish times...")
    late_times = calculate_late_times(df, early_times, project_duration)

    # Calculate slack
    print("Calculating slack times...")
    slack = calculate_slack(early_times, late_times)

    # Find critical path
    critical_path = find_critical_path(df, slack)

    # Print results
    print_results(df, early_times, late_times, slack, critical_path)

    # Visualize schedule
    print("\nGenerating schedule visualization...")
    fig = visualize_schedule(df, early_times, late_times, slack)

    mo.mpl.interactive(fig)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
