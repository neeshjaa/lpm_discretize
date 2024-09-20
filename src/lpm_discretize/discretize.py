import numbers
import numpy as np
import polars as pl
import string

_QUALITATIVE_LABELS = {
    2: ["low", "high"],
    3: ["low", "medium", "high"],
    4: ["very low", "low", "high", "very high"],
    5: ["very low", "low", "medium", "high", "very high"],
}


def _prefix_category_name(idx, total_number_categories):
    """
    Description:
        Create a prefix for readable category name based on an index and
        the total number of categories.

    Parameters:
        - idx (int):  will translate to a letter.
        - total_number_categories:  decides on whether to add qualitiative
          labels like "high"/"low". Needs to be larger than 1. If larger than
          5, no labels are supplied.
    Returns:
        A string that will be used as a prefix for categorical.
    """
    assert idx is not None
    assert total_number_categories is not None
    assert total_number_categories > 1

    lower_case_letters = string.ascii_lowercase
    # Check if the category-index is larger that 26 and hence if
    # we can assign a letter for enhance readability.
    if idx < len(lower_case_letters):
        letter = f"({lower_case_letters[idx]})"
    else:
        return ""
    if total_number_categories <= 5:
        label = _QUALITATIVE_LABELS[total_number_categories][idx] + " "
    else:
        label = ""
    return f"{letter} {label}"


def _readable_category_name(lower_bound=None, upper_bound=None, decimals=1):
    """
    Description:
        Create a readable category name based on upper and lower bounds.

    Parameters:
    - lower_bound (float): lower bound of the category.
    - upper_bound (float): upper bound of the category.
    - decimals (int): ensure not more than n decimals are printed.

    Returns:
        A string with a readable category name, including an alphabetical index,
        if index is <= 26 (index > 26 should be a rare case for bound-based
        categorization; so having it work - yet slightly less readable seems
        fine).
    """
    assert (lower_bound is not None) or (upper_bound is not None)
    if lower_bound is None:
        return f"(> {upper_bound:.{decimals}f})"
    elif upper_bound is None:
        return f"(≤ {lower_bound:.{decimals}f})"
    else:
        assert lower_bound < upper_bound
        return f"({lower_bound:.{decimals}f} - {upper_bound:.{decimals}f})"


def _is_number(value):
    """None/np.nan robust checking for numerical values."""
    if value is None:
        return False
    if isinstance(value, str):
        return False
    if np.isnan(value):
        return False
    return isinstance(value, numbers.Number)


def get_quantile_based_discretization_function(column, quantiles=4, decimals=1):
    """
    Description:
        Takes a reference column that downn the road needs to be discretized and
        returns a function can be used for that. Relies on Polar's qcut
        discretization but retuns a function that can easily be re-used
        and returns human-readable output.

    Parameters:
    - colummn (Polars series or list):  A sequence
    - quantile (int): The number of quantiles used for discretization (default =
      4).

    Returns:
        A function that can be applied to a column for discretization.
    """
    column = [value for value in column if _is_number(value)]
    assert len(set(column)) > 1
    cutoffs = sorted(
        set(
            [
                quantile["breakpoint"]
                for quantile in pl.Series(column).qcut(quantiles, include_breaks=True)
            ]
        )
    )

    def _discretization_function(value):
        assert not isinstance(value, str)
        if _is_number(value):
            for i, cutoff in enumerate(cutoffs[:-1]):
                if value < cutoff:
                    if i == 0:
                        return _prefix_category_name(
                            i, len(cutoffs)
                        ) + _readable_category_name(
                            lower_bound=cutoff, upper_bound=None, decimals=decimals
                        )
                    else:
                        return _prefix_category_name(
                            i, len(cutoffs)
                        ) + _readable_category_name(
                            lower_bound=cutoffs[i - 1],
                            upper_bound=cutoff,
                            decimals=decimals,
                        )
            return _prefix_category_name(
                len(cutoffs) - 1, len(cutoffs)
            ) + _readable_category_name(
                lower_bound=None, upper_bound=cutoffs[i], decimals=decimals
            )

    return _discretization_function


def discretize_column(column, discretization_function):
    """
    Description:
        Discretize a numerical column

    Parameters:
    - colummn (Polars series or list):  A sequence
    - discretization_function (function): A functoin that processes the numerical values and
                                          discretizes them.

    Returns:
        A new list with discretized values

    Examples:
    >>> discretize_column([1,2,1], lambda x: "yes" if x <= 1 else "no")
        ["yes", "no", "yes"]
    """

    # In order for this function to be useful, it needs to:
    #   1. Fail loudly if a string is supplied.
    #   2. Deal gracefully with None/np.nan.
    def _safe_discretization(value):
        assert not isinstance(value, str)
        if _is_number(value):
            return discretization_function(value)

    return [_safe_discretization(value) for value in column]


def discretize_df(df, discretization_functions):
    """
    Description:
        Discretize all numerical columns in a Polars dataframe.

    Parameters:
    - df (Polars dataframe):  A Polars dataframe for which a subset of columns
                              need to be discretized.
    - discretization_functions (dict):
        - a dict from column name to functions, to apply different
          discretization schemes for different columns.

    Returns:
        A new Polars dataframe, where numerical columns are discretized.

    Examples:
    >>> print(df)
        ┌─────┬─────┬─────┬─────┐
        │ foo ┆ bar ┆ ... ┆ baz │
        │ --- ┆ --- ┆ --- ┆ --- │
        │ f64 ┆ i64 ┆ ... ┆ str │
        ╞═════╪═════╪═════╪═════╡
        │ 1.0 ┆ 8   ┆ ... ┆ x   │
        │ 2.0 ┆ 7   ┆ ... ┆ y   │
        │ 3.0 ┆ 6   ┆ ... ┆ x   │
        │ 4.0 ┆ 5   ┆ ... ┆ x   │
        └─────┴─────┴─────┴─────┘

    >>> discretize(df, discretization_functions={
        "foo": lambda x: "yes" if x <= 1 else "no",
        "bar": lambda x: "ja" if x <= 6 else "nein"})
        ┌───────┬────────┬─────┬─────┐
        │ foo   ┆ bar    ┆ ... ┆ baz │
        │ ---   ┆ ---    ┆ --- ┆ --- │
        │ str   ┆ str    ┆ ... ┆ str │
        ╞═══════╪════════╪═════╪═════╡
        │ "yes" ┆ "nein" ┆ ... ┆ x   │
        │ "no"  ┆ "nein" ┆ ... ┆ y   │
        │ "no"  ┆ "ja"   ┆ ... ┆ x   │
        │ "no"  ┆ "ja"   ┆ ... ┆ x   │
        └───────┴────────┴─────┴─────┘
    """

    def _apply_discretization_where_needed(c):
        """Helper function to make list comprehension below more readable."""
        if c in discretization_functions:
            return discretize_column(df[c], discretization_functions[c])
        else:
            return df[c]

    return pl.DataFrame({c: _apply_discretization_where_needed(c) for c in df.columns})


def _set_quantiles(quantiles, columns):
    """Set default value for number of quantiles safely."""
    if quantiles is None:
        quantiles = 4
    if isinstance(quantiles, int):
        quantiles = {c: quantiles for c in columns}
    elif isinstance(quantiles, dict):
        assert set(quantiles.keys()).issubset(set(columns))
    else:
        raise ValueError(quantiles)
    return quantiles


def discretize_df_quantiles(df, quantiles=4, columns=None, decimals=1):
    """
    Description:
        Discretize numerical columns in a Polars dataframe.

    Parameters:
    - df (Polars dataframe):  A Polars dataframe for which a subset of columns
                              need to be discretized.
    - quantiles (dict | int): Either
                                - an int showing how many quartiles to use
                                - dict from column name to functions, to apply
                                  different discretization schemes for different columns.
    - columns:  A list indicating which columns to discretized.  Defaulting to
                Polar's types.

    Returns:
        A new Polars dataframe, where numerical columns are discretized.

    Examples:
    >>> print(df)
        ┌─────┬─────┬─────┬─────┐
        │ foo ┆ bar ┆ ... ┆ baz │
        │ --- ┆ --- ┆ --- ┆ --- │
        │ f64 ┆ i64 ┆ ... ┆ str │
        ╞═════╪═════╪═════╪═════╡
        │ 1.0 ┆ 8   ┆ ... ┆ x   │
        │ 2.0 ┆ 7   ┆ ... ┆ y   │
        │ 3.0 ┆ 6   ┆ ... ┆ x   │
        │ 4.0 ┆ 5   ┆ ... ┆ x   │
        └─────┴─────┴─────┴─────┘

    >>> discretize_df_quantiles(df, quantiles=4)
        ┌─────────────────────────┬──────────────────────────┬─────┬─────┐
        │ foo                     ┆ bar                      ┆ ... ┆ baz │
        │ ---                     ┆ ---                      ┆ --- ┆ --- │
        │ str                     ┆ str                      ┆ ... ┆ str │
        ╞═════════════════════════╪══════════════════════════╪═════╪═════╡
        │ "(a) Very Low (≤ 1.75)" ┆ "(d) Very High (> 7.25)" ┆ ... ┆ x   │
        │ "(b) Low (1.75 - 2.5)"  ┆ "(c) High (6.5 - 7.25)"  ┆ ... ┆ y   │
        │ "(c) High (2.5 - 3.25)" ┆ "(b) Low (5.75 - 6.5)"   ┆ ... ┆ x   │
        │ "(d) Very High (> 3.25)"┆ "(a) Very Low (≤ 5.75)"  ┆ ... ┆ x   │
        └─────────────────────────┴──────────────────────────┴─────┴─────┘,

    >>> discretize_df_quantiles(df, quantiles=2)
        ┌────────────┬────────────┬─────┬─────┐
        │ foo        ┆ bar        ┆ ... ┆ baz │
        │ ---        ┆ ---        ┆ --- ┆ --- │
        │ str        ┆ str        ┆ ... ┆ str │
        ╞════════════╪════════════╪═════╪═════╡
        │ "(a) low"  ┆ "(b) high" ┆ ... ┆ x   │
        │ "(b) high" ┆ "(b) high" ┆ ... ┆ y   │
        │ "(b) high" ┆ "(a) low"  ┆ ... ┆ x   │
        │ "(b) high" ┆ "(a) low"  ┆ ... ┆ x   │
        └────────────┴────────────┴─────┴─────┘
    """
    # Set default values for the columns we want to discretize.
    if columns is None:
        columns = [c for c in df.columns if df[c].dtype.is_numeric()]

    quantiles = _set_quantiles(quantiles, columns)

    return discretize_df(
        df,
        {
            c: get_quantile_based_discretization_function(df[c], q, decimals)
            for c, q in quantiles.items()
        },
    )


def discretize_quantiles(
    dataframes, reference_idx=0, quantiles=None, columns=None, decimals=1
):
    """
    Description:
        Discretize all numerical columns in a list of dataframes supplied. One
        dataframe in the list is chosen as a reference for empirical
        discretization, e.g. based on quantiles (see parameter reference_idx
        below).

    Parameters:
    - dataframes (list):  A list of Polars dataframes.
    - reference_idx (int): Which of these dataframes should be the reference point for
      empirical discretization (e.g. based on quantiles).
    - quantiles (dict | int): Either
                                - an int showing how many quartiles to use
                                - dict from column name to functions, to apply
                                  different discretization schemes for different columns.
    - columns:  A list indicating which columns to discretized.  Defaulting to
                Polar's types.
      (optional, default: Statistical types are guessed based on Polars' types).

    Returns:
        A list of Polars dataframes, where numerical columns are discretized.

    Examples:
    >>> print(dfs)
        [
            ┌─────┬─────┬─────┬─────┐
            │ foo ┆ bar ┆ ... ┆ baz │
            │ --- ┆ --- ┆ --- ┆ --- │
            │ f64 ┆ i64 ┆ ... ┆ str │
            ╞═════╪═════╪═════╪═════╡
            │ 1.0 ┆ 8   ┆ ... ┆ x   │
            │ 2.0 ┆ 7   ┆ ... ┆ y   │
            │ 3.0 ┆ 6   ┆ ... ┆ x   │
            │ 4.0 ┆ 5   ┆ ... ┆ x   │
            └─────┴─────┴─────┴─────┘,
            ┌─────┬─────┬─────┬─────┐
            │ foo ┆ bar ┆ ... ┆ baz │
            │ --- ┆ --- ┆ --- ┆ --- │
            │ f64 ┆ i64 ┆ ... ┆ str │
            ╞═════╪═════╪═════╪═════╡
            │ 4.0 ┆ 5   ┆ ... ┆ z   │
            │ 2.0 ┆ 7   ┆ ... ┆ y   │
            │ 3.0 ┆ 6   ┆ ... ┆ z   │
            │ 4.0 ┆ 5   ┆ ... ┆ x   │
            └─────┴─────┴─────┴─────┘,
            ...
        ]

    >>> discretize_quantiles(dfs)
        [
            ┌─────────────────────────┬──────────────────────────┬─────┬─────┐
            │ foo                     ┆ bar                      ┆ ... ┆ baz │
            │ ---                     ┆ ---                      ┆ --- ┆ --- │
            │ str                     ┆ str                      ┆ ... ┆ str │
            ╞═════════════════════════╪══════════════════════════╪═════╪═════╡
            │ "(a) Very Low (≤ 1.75)" ┆ "(d) Very High (> 7.25)" ┆ ... ┆ x   │
            │ "(b) Low (1.75 - 2.5)"  ┆ "(c) High (6.5 - 7.25)"  ┆ ... ┆ y   │
            │ "(c) High (2.5 - 3.25)" ┆ "(b) Low (5.75 - 6.5)"   ┆ ... ┆ x   │
            │ "(d) Very High (> 3.25)"┆ "(a) Very Low (≤ 5.75)"  ┆ ... ┆ x   │
            └─────────────────────────┴──────────────────────────┴─────┴─────┘,
            ┌─────────────────────────┬──────────────────────────┬─────┬─────┐
            │ foo                     ┆ bar                      ┆ ... ┆ baz │
            │ ---                     ┆ ---                      ┆ --- ┆ --- │
            │ str                     ┆ str                      ┆ ... ┆ str │
            ╞═════════════════════════╪══════════════════════════╪═════╪═════╡
            │ "(d) Very High (> 3.25)"┆ "(a) Very Low (≤ 5.75)"  ┆ ... ┆ z   │
            │ "(b) Low (1.75 - 2.5)"  ┆ "(c) High (6.5 - 7.25)"  ┆ ... ┆ y   │
            │ "(c) High (2.5 - 3.25)" ┆ "(b) Low (5.75 - 6.5)"   ┆ ... ┆ z   │
            │ "(d) Very High (> 3.25)"┆ "(a) Very Low (≤ 5.75)"  ┆ ... ┆ x   │
            └─────────────────────────┴──────────────────────────┴─────┴─────┘,
        ...
        ]

    >>> discretize_quantiles(dfs, quantiles=2)
        [
            ┌─────────────────────┬─────────────────────┬─────┬─────┐
            │ foo                 ┆ bar                 ┆ ... ┆ baz │
            │ ---                 ┆ ---                 ┆ --- ┆ --- │
            │ str                 ┆ str                 ┆ ... ┆ str │
            ╞═════════════════════╪═════════════════════╪═════╪═════╡
            │ "(a) Low (≤ 2.5)"   ┆ "(b) High (> 6.5 )" ┆ ... ┆ x   │
            │ "(a) Low (≤ 2.5)"   ┆ "(b) High (> 6.5 )" ┆ ... ┆ y   │
            │ "(b) High (> 2.5 )" ┆ "(a) Low (≤ 2.5)"   ┆ ... ┆ x   │
            │ "(b) High (> 2.5 )" ┆ "(a) Low (≤ 2.5)"   ┆ ... ┆ x   │
            └─────────────────────┴─────────────────────┴─────┴─────┘,
            ┌─────────────────────┬─────────────────────┬─────┬─────┐
            │ foo                 ┆ bar                 ┆ ... ┆ baz │
            │ ---                 ┆ ---                 ┆ --- ┆ --- │
            │ str                 ┆ str                 ┆ ... ┆ str │
            ╞═════════════════════╪═════════════════════╪═════╪═════╡
            │ "(b) High (> 2.5 )" ┆ "(a) Low (≤ 2.5)"   ┆ ... ┆ x   │
            │ "(a) Low (≤ 2.5)"   ┆ "(b) High (> 6.5 )" ┆ ... ┆ y   │
            │ "(b) High (> 2.5 )" ┆ "(a) Low (≤ 2.5)"   ┆ ... ┆ x   │
            │ "(b) High (> 2.5 )" ┆ "(a) Low (≤ 2.5)"   ┆ ... ┆ x   │
            └─────────────────────┴─────────────────────┴─────┴─────┘,
        ...
        ]

    >>> discretize_quantiles(dfs, quantiles={"foo":4, "bar":2})
        [
            ┌─────────────────────────┬─────────────────────┬─────┬─────┐
            │ foo                     ┆ bar                 ┆ ... ┆ baz │
            │ ---                     ┆ ---                 ┆ --- ┆ --- │
            │ str                     ┆ str                 ┆ ... ┆ str │
            ╞═════════════════════════╪═════════════════════╪═════╪═════╡
            │ "(a) Very Low (≤ 1.75)" ┆ "(b) High (> 6.5 )" ┆ ... ┆ x   │
            │ "(b) Low (1.75 - 2.5)"  ┆ "(b) High (> 6.5 )" ┆ ... ┆ y   │
            │ "(c) High (2.5 - 3.25)" ┆ "(a) Low (≤ 2.5)"   ┆ ... ┆ x   │
            │ "(d) Very High (> 3.25)"┆ "(a) Low (≤ 2.5)"   ┆ ... ┆ x   │
            └─────────────────────────┴─────────────────────┴─────┴─────┘,
            ┌─────────────────────────┬─────────────────────┬─────┬─────┐
            │ foo                     ┆ bar                 ┆ ... ┆ baz │
            │ ---                     ┆ ---                 ┆ --- ┆ --- │
            │ str                     ┆ str                 ┆ ... ┆ str │
            ╞═════════════════════════╪═════════════════════╪═════╪═════╡
            │ "(d) Very High (> 3.25)"┆ "(a) Low (≤ 2.5)"   ┆ ... ┆ x   │
            │ "(b) Low (1.75 - 2.5)"  ┆ "(b) High (> 6.5 )" ┆ ... ┆ y   │
            │ "(c) High (2.5 - 3.25)" ┆ "(a) Low (≤ 2.5)"   ┆ ... ┆ x   │
            │ "(d) Very High (> 3.25)"┆ "(a) Low (≤ 2.5)"   ┆ ... ┆ x   │
            └─────────────────────────┴─────────────────────┴─────┴─────┘,
        ...
        ]
    """
    # Set default values for the columns we want to discretize.
    if columns is None:
        columns = [
            c
            for c in dataframes[reference_idx].columns
            if dataframes[reference_idx][c].dtype.is_numeric()
        ]
    # Check that the columns agree.
    for df in dataframes[1:]:
        assert set(df.columns) == set(dataframes[0].columns)

    quantiles = _set_quantiles(quantiles, columns)
    discretization_functions = {
        column_name: get_quantile_based_discretization_function(
            dataframes[reference_idx][column_name],
            quantiles=quantile_number,
            decimals=decimals,
        )
        for column_name, quantile_number in quantiles.items()
    }
    return [discretize_df(df, discretization_functions) for df in dataframes]
