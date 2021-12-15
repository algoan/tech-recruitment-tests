"""This module contains regularity detection functions."""
import pandas as pd
import numpy as np

def __get_intervals(transactions_dates):
    """
    Compute periods between dates in array dates.

    Parameters
    ----------
    transactions_dates : PandasSeries
        transactions dates and they should be a pandas date format

    Returns
    -------
    list[int]
        The periods bewteen dates in array
    """
    return list(map(lambda x: x.days, np.diff(transactions_dates)))


def __frequency(serie):
    """
    Match a mean range with a name

    Parameters
    ----------
    serie : PandasSeries
        The mean serie column

    Returns
    -------
    PandasSeries
        A serie with matched ranges names
    """
    bins = pd.IntervalIndex.from_tuples(
        [(6, 8), (8, 12), (12, 16), (16, 25), (25, 35), (35, 55), (55, 65)],
        closed="left",
    )
    cats = ["WEEK", "W-F", "FORTNIGHT", "F-M", "MONTH", "M-2M", "2MONTHS"]
    return np.array(cats)[pd.cut(serie, bins=bins).cat.codes]


def detect_regular_transactions(transactions):
    """
    Return a boolean for each transaction indicating if it belongs to a regular group.

    Entry
    ------
    transactions <pd.DataFrame>
        date <datetime.date>
        amount <float>
        description <str>
        type <str>
        category <str>
        group_id <str>

    ------
    Returns copied transactions dataframe with 2 new columns
        is_regular <pd.Series> <bool>
        frequency <pd.Series> <str>
    """
    # initialize
    table = transactions.copy()
    table["regularity"] = "N"
    table["frequency"] = None

    # compute mean interval
    mean_interval = (
        table.query("amount > 0")
        .groupby("group_id")["date"]
        .transform(lambda x: np.mean(__get_intervals(x)) if len(x) > 2 else None)
    ).dropna()

    # compute frequency based on mean interval
    frequency = pd.Series(__frequency(mean_interval), index=mean_interval.index)
    table.loc[frequency.index, "frequency"] = frequency
    table["regularity"] = table.frequency.notnull().apply(
        lambda x: "Y" if x else "N"
    )

    return table
