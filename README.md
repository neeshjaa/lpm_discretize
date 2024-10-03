# lpm-discretize

This repo was forked from [here](https://github.com/inferenceql/lpm.discretize).

## Overview

A library for taking Polars data frames producing new data frames where all
numerical columns have been discretized.

## Disclaimer
This is pre-alpha software. We are currently testing it in real-world scenarios. In its present state, we discourage users from trying it.

## Installation

This library is packaged with [Poetry](https://python-poetry.org/). Add this
line to your `pyproject.toml` file:
```toml
lpm-discretize = {git = "https://github.com/neeshjaa/lpm_discretize.git", branch = "main"}
```

## Usage

Users can discretize a single dataframe or a list of dataframes.

If multiple dataframes are supplied, one will be used as a reference point. This
is necessary for example when quantile based discretization is applied.

They can supply custom functions for discretization or use quantile-based discretization.

### Using LPM.discretize as a Python library

#### Discretize a single dataframe based on quantiles
Tak a single dataframe and discretize numerical columns based on a quantiles:
```python
# Get dependencies.
import polars as pl
from lpm_discretize import discretize_df_quantiles

# Read a csv files.
df = pl.read_csv("data.csv")

# Discretize this dataframe; all columns are discretized based on 4 quantiles. 
df_discretized =  discretize_df_quantiles(df, quantiles=4)
```
Users can list which columns will be discretized. Let's discretize only `foo` and
`bar`.
```python
df_discretized =  discretize_df_quantiles(df, quantiles=4, columns=["foo", "bar"])
```

Note that the `quantiles` argument here is overloaded. Users can supply an int to set the
number of quantiles for every column, or they supply a dictionary.
```python
df_discretized =  discretize_df_quantiles(df, quantiles={"foo": 4, "bar": 2}))
```

#### Discretize a single dataframe based on a map columns->discretization-functions

Users can supply their own discretization functions as dictionaries.
```python
from lpm_discretize import discretize_df
discretize(df, discretization_functions={"foo": lambda x: "yes" if x <= 1 else "no", "bar": lambda x: "ja" if x <= 6 else "nein"})
```

Of course, this can be wrapped into a list comprehension discretizing multiple dataframes.

#### Discretizing multiple dataframes based on quantiles

```python
import polars as pl
from lpm_discretize import discretize_df_quantiles

# Read three csv files.
df_a = pl.read_csv("real-data.csv")
df_b = pl.read_csv("synthetic-data-lpm.csv")
df_c = pl.read_csv("synthetic-data-baseline.csv")

# Get discretized version of the dataframes above. Use Polar's types do decide what to discretize.
# By default, this discretizes with based on quartiles in `df_a`.
df_a_discretized, df_b_discretized, df_c_discretized =  discretize_quantiles([df_a, df_b, df_c], quantiles=4)
```

```python
Below use a list of columns to discretize
df_a_discretized, df_b_discretized, df_c_discretized =  discretize_quantiles(
    [df_a, df_b, df_c],
    quantiles=4,
    columns=["foo", "bar"]
)
```
See docstrings for other usage patterns.

## Test

Tests can be run with Poetry

```shell
poetry run pytest tests/ -vvv
```
