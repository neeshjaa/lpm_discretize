import numpy as np
import polars as pl
import pytest

from inspect import isfunction
from polars.testing import assert_frame_equal
from polars.testing import assert_frame_not_equal

from lpm_discretize import get_quantile_based_discretization_function
from lpm_discretize import _is_number
from lpm_discretize import _prefix_category_name
from lpm_discretize import _readable_category_name
from lpm_discretize import discretize
from lpm_discretize import discretize_column
from lpm_discretize import discretize_df
from lpm_discretize import discretize_df_quantiles
from lpm_discretize import discretize_quantiles


def _assert_raised_AssertionError(f, *args):
    """Helper function testing that an AssertionError is raised."""
    with pytest.raises(AssertionError) as exc_info:
        f(*args)
    assert exc_info.type == AssertionError


def test_prefix_category_name_args_None():
    _assert_raised_AssertionError(_prefix_category_name, 42, None)
    _assert_raised_AssertionError(_prefix_category_name, None, 42)


def test_prefix_category_name_args_one_category():
    _assert_raised_AssertionError(_prefix_category_name, 0, 1)


def test_prefix_category_name_empty_prefix():
    # Set the total number of categories larger than threshold for supplying
    # qualitative category names.
    total_number_categories = (
        27  # more categories than there are letters in the alphabet
    )
    assert (
        _prefix_category_name(total_number_categories - 1, total_number_categories)
        == ""
    )


def test_prefix_category_name_letters():
    # Set the total number of categories larger than threshold for supplying
    # qualitative category names.
    total_number_categories = 6
    assert _prefix_category_name(0, total_number_categories) == "(a) "
    assert _prefix_category_name(1, total_number_categories) == "(b) "
    assert _prefix_category_name(2, total_number_categories) == "(c) "


def test_prefix_category_name_letters_and_2_categories():
    total_number_categories = 2
    assert _prefix_category_name(0, total_number_categories) == "(a) low "
    assert _prefix_category_name(1, total_number_categories) == "(b) high "


def test_prefix_category_name_letters_and_3_categories():
    total_number_categories = 3
    assert _prefix_category_name(0, total_number_categories) == "(a) low "
    assert _prefix_category_name(1, total_number_categories) == "(b) medium "
    assert _prefix_category_name(2, total_number_categories) == "(c) high "


def test_prefix_category_name_letters_and_4_categories():
    total_number_categories = 4
    assert _prefix_category_name(0, total_number_categories) == "(a) very low "
    assert _prefix_category_name(1, total_number_categories) == "(b) low "
    assert _prefix_category_name(2, total_number_categories) == "(c) high "
    assert _prefix_category_name(3, total_number_categories) == "(d) very high "


def test_prefix_category_name_letters_and_5_categories():
    total_number_categories = 5
    assert _prefix_category_name(0, total_number_categories) == "(a) very low "
    assert _prefix_category_name(1, total_number_categories) == "(b) low "
    assert _prefix_category_name(2, total_number_categories) == "(c) medium "
    assert _prefix_category_name(3, total_number_categories) == "(d) high "
    assert _prefix_category_name(4, total_number_categories) == "(e) very high "


def test_readable_category_name_both_None():
    _assert_raised_AssertionError(_readable_category_name, None, None)


def test_readable_category_name_lower_higher_upper():
    _assert_raised_AssertionError(_readable_category_name, 42, 17)


@pytest.mark.parametrize(
    "upper, lower, expected_result",
    [
        (
            17.0,
            None,
            "(≤ 17.0)",
        ),
        (
            None,
            17.0,
            "(> 17.0)",
        ),
        (
            17.0,
            42.0,
            "(17.0 - 42.0)",
        ),
        # Now use ints
        (
            17,
            None,
            "(≤ 17.0)",
        ),
        (
            None,
            17,
            "(> 17.0)",
        ),
        (
            17,
            42,
            "(17.0 - 42.0)",
        ),
    ],
)
def test_readable_category_name_without_specifying_decimals(
    upper, lower, expected_result
):
    _readable_category_name(upper, lower) == expected_result


@pytest.mark.parametrize(
    "upper, lower, decimals, expected_result",
    [
        (
            17.00,
            None,
            2,
            "(≤ 17.00)",
        ),
        (
            None,
            17.00,
            2,
            "(> 17.00)",
        ),
        (
            17.00,
            42.00,
            2,
            "(17.00 - 42.00)",
        ),
        (
            17.000,
            None,
            3,
            "(≤ 17.000)",
        ),
        (
            None,
            17.000,
            3,
            "(> 17.000)",
        ),
        (
            17.000,
            42.000,
            3,
            "(17.000 - 42.000)",
        ),
    ],
)
def test_readable_category_name_with_specifying_decimals(
    upper, lower, decimals, expected_result
):
    _readable_category_name(upper, lower, decimals=decimals) == expected_result


@pytest.mark.parametrize("value", [None, np.nan, "x", "abc"])
def test_is_not_number(value):
    assert not _is_number(value)


@pytest.mark.parametrize("value", [0, 1, 42.0, -1])
def test_is_number(value):
    assert _is_number(value)


@pytest.mark.parametrize(
    "invalid_column",
    [
        [],
        [42],
        [42, 42],
        [42, None, 42],
        [42, np.nan, 42],
    ],
)
def test_get_quantile_based_discretization_function_invalid_input(invalid_column):
    _assert_raised_AssertionError(
        get_quantile_based_discretization_function, invalid_column
    )


def test_get_quantile_based_discretization_function_smoke():
    assert isfunction(get_quantile_based_discretization_function(range(3)))


@pytest.mark.parametrize("n_quantiles", range(2, 12))
def test_get_quantile_based_discretization_function_lower(n_quantiles):
    f = get_quantile_based_discretization_function(
        range(1, n_quantiles + 2), quantiles=n_quantiles
    )
    if n_quantiles < 4:
        assert f(-1) == "(a) low (≤ 2.0)"
    elif n_quantiles < 6:
        assert f(-1) == "(a) very low (≤ 2.0)"
    else:
        assert f(-1) == "(a) (≤ 2.0)"


@pytest.mark.parametrize("n_quantiles_upper", range(6, 16))
def test_get_quantile_based_discretization_function_upper(n_quantiles_upper):
    f = get_quantile_based_discretization_function(
        range(1, n_quantiles_upper + 2), quantiles=n_quantiles_upper
    )
    assert f(16)[4:] == f"(> {float(n_quantiles_upper)})"


@pytest.mark.parametrize("missing_value", [np.nan, None])
def test_get_quantile_based_discretization_function_in_presence_of_NaN(missing_value):
    f = get_quantile_based_discretization_function([1, 2, 3], quantiles=2)
    assert f(missing_value) is None


def test_get_quantile_based_discretization_function_in_presence_of_str_value():
    f = get_quantile_based_discretization_function([1, 2, 3, 4])
    _assert_raised_AssertionError(f, "foo")


def test_get_quantile_based_discretization_function_quartiles_spot_check():
    f = get_quantile_based_discretization_function(range(5), quantiles=4)
    assert f(2.5) == "(c) high (2.0 - 3.0)"


def test_get_quantile_based_discretization_function_6_split_spot_check():
    f = get_quantile_based_discretization_function(range(10), quantiles=5)
    assert f(5) == "(c) medium (3.6 - 5.4)"


def test_get_quantile_based_discretization_function_4_split_spot_check():
    column = range(1, 6)
    f = get_quantile_based_discretization_function(column)
    assert f(column[-1]) == "(d) very high (> 4.0)"


DISCRETIZE_BIN = lambda x: "yes" if x <= 1 else "no"


def test_discretize_column_binary():
    assert discretize_column([1, 2, 1], DISCRETIZE_BIN) == ["yes", "no", "yes"]


def test_discretize_column_binary_None():
    assert discretize_column([1, 2, None, 1], DISCRETIZE_BIN) == [
        "yes",
        "no",
        None,
        "yes",
    ]


def test_discretize_column_binary_np_nan():
    assert discretize_column([1, 2, np.nan, 1], DISCRETIZE_BIN) == [
        "yes",
        "no",
        None,
        "yes",
    ]


def test_discretize_column_with_get_quantile_based_discretization_function():
    column = range(1, 6)
    f = get_quantile_based_discretization_function(column)
    assert discretize_column(column, f) == [
        "(a) very low (≤ 2.0)",
        "(b) low (2.0 - 3.0)",
        "(c) high (3.0 - 4.0)",
        "(d) very high (> 4.0)",
        "(d) very high (> 4.0)",  # that looks weird but is correct. See spot check above, too.
    ]


def test_discretize_column_non_mutable():
    """Ensure we don't mutate the original column."""
    column = [1, 2]
    assert discretize_column(column, DISCRETIZE_BIN) != column


def test_discretize_df():
    df = pl.DataFrame(
        {"foo": np.linspace(1, 4, 4), "bar": range(5, 9), "baz": ["x", "y", "x", "y"]}
    )
    df_expected = pl.DataFrame(
        {
            "foo": ["yes", "no", "no", "no"],
            "bar": ["ja", "ja", "nein", "nein"],
            "baz": ["x", "y", "x", "y"],
        }
    )
    discretized_df = discretize_df(
        df,
        discretization_functions={
            "foo": DISCRETIZE_BIN,
            "bar": lambda x: "ja" if x <= 6 else "nein",
        },
    )
    assert_frame_equal(discretized_df, df_expected)


def test_discretize_df_non_mutable():
    df = pl.DataFrame({"foo": [1, 2]})
    discretized_df = discretize_df(df, discretization_functions={"foo": DISCRETIZE_BIN})
    assert_frame_not_equal(discretized_df, df)


@pytest.mark.parametrize("df_args", [{"foo": ["a"]}, {"bar": range(6)}, None])
def test_discretize_df_null_op(df_args):
    df = pl.DataFrame(df_args)
    discretized_df = discretize_df(df, discretization_functions={})
    assert_frame_equal(discretized_df,df)


# Some constructs used multiple times below.
_numeric_column = range(1, 6)
_discrete_column = ["a"] * len(_numeric_column)
_df_input = pl.DataFrame({"foo": _numeric_column, "bar": _discrete_column})

_df_expected = pl.DataFrame(
    {
        "foo": [
            "(a) very low (≤ 2.0)",
            "(b) low (2.0 - 3.0)",
            "(c) high (3.0 - 4.0)",
            "(d) very high (> 4.0)",
            "(d) very high (> 4.0)",
        ],
        "bar": _discrete_column,
    }
)


def test_discretize_df_supplying_quantiles():
    discretized_df = discretize_df(
        _df_input,
        discretization_functions={
            "foo": get_quantile_based_discretization_function(_numeric_column)
        },
    )
    assert_frame_equal(discretized_df, _df_expected)


_quantiles_foo = [
    {"foo": 4},  # explicit dict.
    4,  # should be turned into dict.
    None,  # default.
]


@pytest.mark.parametrize("quantiles_foo", _quantiles_foo)
def test_discretize_df_quantiles(quantiles_foo):
    discretized_df = discretize_df_quantiles(_df_input, quantiles=quantiles_foo)
    assert_frame_equal(discretized_df, _df_expected)


@pytest.mark.parametrize("quantiles_foo", _quantiles_foo)
def test_discretize_df_quantiles_set_columns(quantiles_foo):
    discretized_df = discretize_df_quantiles(
        _df_input, quantiles=quantiles_foo, columns=["foo"]
    )
    assert_frame_equal(discretized_df, _df_expected)


def test_discretize_df_quantiles_non_mutable():
    discretized_df = discretize_df_quantiles(_df_input)
    assert_frame_not_equal(discretized_df, _df_input)


def test_discretize_df_quantiles_non_existing_columns():
    _assert_raised_AssertionError(discretize_df_quantiles, _df_input, {"quagga": 2})


@pytest.mark.parametrize("quantiles_foo", _quantiles_foo)
def test_discretize_quantiles_smoke(quantiles_foo):
    discretized_df1, discretized_df2 = discretize_quantiles(
        [_df_input, _df_input], quantiles=quantiles_foo
    )
    assert isinstance(discretized_df1, pl.DataFrame) and isinstance(
        discretized_df1, pl.DataFrame
    )


_dfs = [
    pl.DataFrame(
        {
            "foo": range(1, 6),
            "bar": range(5, 10),
            "baz": _discrete_column,
        }
    ),
    pl.DataFrame(
        {
            "foo": [5, 2, 3, 4],
            "bar": [5, 7, 6, 8],
            "baz": _discrete_column[:-1],
        }
    ),
]


def test_discretize_quantiles_spot_check_int_quartiles():
    discretized_dfs = discretize_quantiles(_dfs)
    expected = [
        pl.DataFrame(
            {
                "foo": [
                    "(a) very low (≤ 2.0)",
                    "(b) low (2.0 - 3.0)",
                    "(c) high (3.0 - 4.0)",
                    "(d) very high (> 4.0)",
                    "(d) very high (> 4.0)",
                ],
                "bar": [
                    "(a) very low (≤ 6.0)",
                    "(b) low (6.0 - 7.0)",
                    "(c) high (7.0 - 8.0)",
                    "(d) very high (> 8.0)",
                    "(d) very high (> 8.0)",
                ],
                "baz": _discrete_column,
            }
        ),
        pl.DataFrame(
            {
                "foo": [
                    "(d) very high (> 4.0)",
                    "(b) low (2.0 - 3.0)",
                    "(c) high (3.0 - 4.0)",
                    "(d) very high (> 4.0)",
                ],
                "bar": [
                    "(a) very low (≤ 6.0)",
                    "(c) high (7.0 - 8.0)",
                    "(b) low (6.0 - 7.0)",
                    "(d) very high (> 8.0)",
                ],
                "baz": _discrete_column[:-1],
            }
        ),
    ]
    assert_frame_equal(discretized_dfs[0], expected[0])
    assert_frame_equal(discretized_dfs[1], expected[1])


def test_discretize_quantiles_spot_check_quantiles_per_col():
    discretized_dfs = discretize_quantiles(_dfs, quantiles={"foo": 4, "bar": 2})
    expected = [
        pl.DataFrame(
            {
                "foo": [
                    "(a) very low (≤ 2.0)",
                    "(b) low (2.0 - 3.0)",
                    "(c) high (3.0 - 4.0)",
                    "(d) very high (> 4.0)",
                    "(d) very high (> 4.0)",
                ],
                "bar": [
                    "(a) low (≤ 7.0)",
                    "(a) low (≤ 7.0)",
                    "(b) high (> 7.0)",
                    "(b) high (> 7.0)",
                    "(b) high (> 7.0)",
                ],
                "baz": _discrete_column,
            }
        ),
        pl.DataFrame(
            {
                "foo": [
                    "(d) very high (> 4.0)",
                    "(b) low (2.0 - 3.0)",
                    "(c) high (3.0 - 4.0)",
                    "(d) very high (> 4.0)",
                ],
                "bar": [
                    "(a) low (≤ 7.0)",
                    "(b) high (> 7.0)",
                    "(a) low (≤ 7.0)",
                    "(b) high (> 7.0)",
                ],
                "baz": _discrete_column[:-1],
            }
        ),
    ]
    assert_frame_equal(discretized_dfs[0], expected[0])
    assert_frame_equal(discretized_dfs[1], expected[1])
