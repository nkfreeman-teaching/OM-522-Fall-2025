"""
Critical Path Method (CPM) utilities for project scheduling.

This module provides utilities to analyze project schedules using the
Critical Path Method. It calculates early/late start/finish times for
each task in a project network.
"""

import polars as pl
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def calculate_cpm(df: pl.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Calculate Critical Path Method metrics for a project schedule.

    Args:
        df: A Polars DataFrame with three columns:
            - "current_machine_job" (str): Task identifier
            - "predecessors" (list): List of predecessor task identifiers
            - "pij" (int): Task duration

    Returns:
        A dictionary where keys are task names (current_machine_job values)
        and values are dictionaries containing:
            - "early_start": Earliest time task can start
            - "early_finish": Earliest time task can finish
            - "late_start": Latest time task can start without delaying project
            - "late_finish": Latest time task can finish without delaying project

    Example:
        >>> df = pl.DataFrame({
        ...     "current_machine_job": ["A", "B", "C"],
        ...     "predecessors": [[], ["A"], ["A", "B"]],
        ...     "pij": [5, 3, 4]
        ... })
        >>> result = calculate_cpm(df)
        >>> result["A"]
        {'early_start': 0, 'early_finish': 5, 'late_start': 0, 'late_finish': 5}
    """

    # Step 1: Calculate early times (forward pass)
    early_times = _calculate_early_times(df)

    # Step 2: Determine project duration
    project_duration = max(ef for es, ef in early_times.values())

    # Step 3: Calculate late times (backward pass)
    late_times = _calculate_late_times(df, early_times, project_duration)

    # Step 4: Format results into the required dictionary structure
    result = {}
    for row in df.iter_rows(named=True):
        task = row["current_machine_job"]
        es, ef = early_times[task]
        ls, lf = late_times[task]

        result[task] = {
            "early_start": es,
            "early_finish": ef,
            "late_start": ls,
            "late_finish": lf
        }

    return result


def _calculate_early_times(df: pl.DataFrame) -> Dict[str, tuple]:
    """
    Calculate Early Start (ES) and Early Finish (EF) times for all tasks.

    The early start is the earliest a task can begin (after all predecessors finish).
    The early finish is early start + duration.

    Args:
        df: DataFrame with project data

    Returns:
        Dictionary mapping task name to (ES, EF) tuple
    """
    early_times = {}

    # Create a mapping of task to its data
    task_data = {}
    for row in df.iter_rows(named=True):
        task = row["current_machine_job"]
        task_data[task] = {
            "duration": row["pij"],
            "predecessors": row["predecessors"] if row["predecessors"] else []
        }

    # Process tasks in topological order
    # Keep track of tasks that still need to be processed
    remaining_tasks = set(task_data.keys())

    while remaining_tasks:
        # Find tasks whose predecessors have all been processed
        ready_tasks = []
        for task in remaining_tasks:
            predecessors = task_data[task]["predecessors"]
            if all(pred in early_times for pred in predecessors):
                ready_tasks.append(task)

        # If no tasks are ready, we have a circular dependency
        if not ready_tasks:
            raise ValueError("Circular dependency detected in task predecessors")

        # Process all ready tasks
        for task in ready_tasks:
            duration = task_data[task]["duration"]
            predecessors = task_data[task]["predecessors"]

            # Calculate Early Start (ES)
            if not predecessors:
                # No predecessors means this task can start at time 0
                es = 0
            else:
                # ES is the maximum EF of all predecessors
                es = max(early_times[pred][1] for pred in predecessors)

            # Calculate Early Finish (EF)
            ef = es + duration

            early_times[task] = (es, ef)
            remaining_tasks.remove(task)

    return early_times


def _calculate_late_times(
    df: pl.DataFrame,
    early_times: Dict[str, tuple],
    project_duration: int
) -> Dict[str, tuple]:
    """
    Calculate Late Start (LS) and Late Finish (LF) times for all tasks.

    The late finish is the latest a task can finish without delaying the project.
    The late start is late finish - duration.

    Args:
        df: DataFrame with project data
        early_times: Dictionary of early start/finish times
        project_duration: Total project duration (maximum EF)

    Returns:
        Dictionary mapping task name to (LS, LF) tuple
    """
    late_times = {}

    # Create a mapping of which tasks depend on each task (reverse dependencies)
    # and store task durations
    successors = defaultdict(list)
    task_durations = {}

    for row in df.iter_rows(named=True):
        task = row["current_machine_job"]
        task_durations[task] = row["pij"]
        predecessors = row["predecessors"] if row["predecessors"] else []
        for pred in predecessors:
            successors[pred].append(task)

    # Process tasks in reverse topological order
    # Start with tasks that have no successors or whose successors have been processed
    remaining_tasks = set(task_durations.keys())

    while remaining_tasks:
        # Find tasks whose successors have all been processed
        ready_tasks = []
        for task in remaining_tasks:
            task_successors = successors.get(task, [])
            if all(succ in late_times for succ in task_successors):
                ready_tasks.append(task)

        # If no tasks are ready, we have an issue (shouldn't happen if forward pass worked)
        if not ready_tasks:
            raise ValueError("Unable to process tasks in reverse order")

        # Process all ready tasks
        for task in ready_tasks:
            duration = task_durations[task]

            # Calculate Late Finish (LF)
            if task not in successors or not successors[task]:
                # No successors means this is a final task
                lf = project_duration
            else:
                # LF is the minimum LS of all successors
                lf = min(late_times[succ][0] for succ in successors[task])

            # Calculate Late Start (LS)
            ls = lf - duration

            late_times[task] = (ls, lf)
            remaining_tasks.remove(task)

    return late_times


def minimize_maximum_lateness(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Solve the single machine scheduling problem to minimize maximum lateness.

    Uses a dynamic earliest due date (EDD) rule to generate the job sequence,
    considering release times. Jobs are scheduled as early as possible based on
    their availability and due dates.

    Args:
        df: A Polars DataFrame with four columns:
            - "job" (str): Job identifier
            - "pj" (float): Job processing time
            - "rj" (float): Job release time
            - "dj" (float): Job due date

    Returns:
        A dictionary containing:
            - "sequence": List of job IDs in the scheduled order
            - "max_lateness": The maximum lateness value across all jobs

    Example:
        >>> df = pl.DataFrame({
        ...     "job": ["J1", "J2", "J3"],
        ...     "pj": [3, 2, 4],
        ...     "rj": [0, 1, 0],
        ...     "dj": [5, 8, 10]
        ... })
        >>> result = minimize_maximum_lateness(df)
        >>> result["max_lateness"]
    """
    # Convert DataFrame to list of dictionaries for easier manipulation
    jobs = df.to_dicts()
    scheduled = []
    current_time = 0
    available_jobs = []

    while len(scheduled) < len(jobs):
        # Find all jobs that are available (released) at current_time
        for job in jobs:
            if job not in scheduled and job["rj"] <= current_time:
                if job not in available_jobs:
                    available_jobs.append(job)

        # If no jobs are available, advance time to the next release time
        if not available_jobs:
            next_release = min(job["rj"] for job in jobs if job not in scheduled)
            current_time = next_release
            continue

        # Select job with earliest due date among available jobs
        selected_job = min(available_jobs, key=lambda j: j["dj"])
        available_jobs.remove(selected_job)
        scheduled.append(selected_job)

        # Update current time
        current_time += selected_job["pj"]

    # Calculate completion times and lateness for each job
    current_time = 0
    max_lateness = float('-inf')
    job_details = []

    for job in scheduled:
        # Job starts at max(current_time, release_time)
        start_time = max(current_time, job["rj"])
        completion_time = start_time + job["pj"]
        lateness = completion_time - job["dj"]
        max_lateness = max(max_lateness, lateness)

        job_details.append({
            "job": job["job"],
            "start": start_time,
            "completion": completion_time,
            "lateness": lateness
        })

        current_time = completion_time

    sequence = [job["job"] for job in scheduled]

    return {
        "sequence": sequence,
        "max_lateness": max_lateness,
        "job_details": job_details
    }


def parse_machine_job(machine_job_str: str) -> Tuple[str, str]:
    """
    Parse the machine,job string into separate components.

    Args:
        machine_job_str: String in format "machine,job" (e.g., "3,2")

    Returns:
        Tuple of (machine, job) as strings
    """
    parts = machine_job_str.split(',')
    return parts[0], parts[1]


def create_gantt_chart(df: pl.DataFrame) -> Tuple[Any, Any]:
    """
    Create a Gantt chart from the solution data.

    Args:
        df: Polars DataFrame with current_machine_job, predecessors, and pij columns
        output_file: Path to save the output image

    Returns:
        Tuple of (figure, axes) matplotlib objects
    """
    # Calculate CPM to get early start times
    cpm_results = calculate_cpm(df)

    # Parse data and organize by machine
    machine_jobs = {}  # machine -> list of (job, start_time, duration)

    for row in df.iter_rows(named=True):
        machine_job = row['current_machine_job']
        machine, job = parse_machine_job(machine_job)

        early_start = cpm_results[machine_job]['early_start']
        duration = row['pij']

        if machine not in machine_jobs:
            machine_jobs[machine] = []

        machine_jobs[machine].append({
            'job': job,
            'start': early_start,
            'duration': duration,
            'end': early_start + duration
        })

    # Sort machines for consistent display
    machines = sorted(machine_jobs.keys(), key=lambda x: int(x))

    # Create the Gantt chart
    fig, ax = plt.subplots(figsize=(12, max(6, len(machines) * 0.8)))

    # Color palette for jobs
    colors = plt.cm.tab10.colors

    # Plot each machine's jobs
    for i, machine in enumerate(machines):
        jobs = machine_jobs[machine]

        for job_info in jobs:
            # Draw the bar
            bar = ax.barh(
                i,
                job_info['duration'],
                left=job_info['start'],
                height=0.6,
                color=colors[int(job_info['job']) % len(colors)],
                edgecolor='black',
                linewidth=1.5
            )

            # Annotate with job number
            ax.text(
                job_info['start'] + job_info['duration'] / 2,
                i,
                f"J{job_info['job']}",
                ha='center',
                va='center',
                fontweight='bold',
                fontsize=10,
                color='white' if int(job_info['job']) % len(colors) < 5 else 'black'
            )

    # Customize the plot
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f'Machine {m}' for m in machines])
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Machine', fontsize=12, fontweight='bold')
    ax.set_title('Gantt Chart - Job Scheduling by Machine\n(Early Start Times)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add grid for better readability
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set x-axis to start at 0
    ax.set_xlim(left=0)

    # Tight layout
    plt.tight_layout()

    return fig, ax
