import numpy as np
import pandas as pd
from quokkas.core.frames.dataframe import DataFrame
from quokkas.core.generic.algos import unique, concat


def test_unique(df_random):
    uniques = unique(df_random.iloc[:, 0])
    assert isinstance(uniques, np.ndarray)


def test_concat(df_random, df_random_unbalanced):
    concatenated = concat((df_random, df_random_unbalanced), axis=1)
    assert isinstance(concatenated, DataFrame)
    assert concatenated.shape == (df_random.shape[0], df_random.shape[1] + df_random_unbalanced.shape[1])

    concatenated_series = concat((df_random.iloc[:, 0], df_random_unbalanced.iloc[:, 1]))
    assert isinstance(concatenated_series, pd.Series)
    assert concatenated_series.shape == (df_random.shape[0] * 2,)
