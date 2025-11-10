"""
Critical Path Method (CPM) utilities for project scheduling.

This module provides utilities to analyze project schedules using the
Critical Path Method. It calculates early/late start/finish times for
each task in a project network.
"""

import polars as pl
from collections import defaultdict
from typing import Dict, List, Any


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

    # Process tasks in order
    for row in df.iter_rows(named=True):
        task = row["current_machine_job"]
        duration = row["pij"]
        predecessors = row["predecessors"] if row["predecessors"] else []

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
    successors = defaultdict(list)
    for row in df.iter_rows(named=True):
        task = row["current_machine_job"]
        predecessors = row["predecessors"] if row["predecessors"] else []
        for pred in predecessors:
            successors[pred].append(task)

    # Process tasks in reverse order
    tasks = df["current_machine_job"].to_list()
    for task in reversed(tasks):
        duration = df.filter(pl.col("current_machine_job") == task)["pij"][0]

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

    return late_times
