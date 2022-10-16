import datetime
import math
import os

import numpy as np
import pandas as pd
import pytest

from quokkas.core.frames.dataframe import DataFrame
np.random.seed(0)

@pytest.fixture
def df():
    return DataFrame(np.ones((10000, 10)), columns=['col_' + str(x) for x in range(10)])


@pytest.fixture
def df_mini():
    return DataFrame(np.ones((10, 10)), columns=['col_' + str(x) for x in range(10)])


@pytest.fixture
def df_pipelined():
    df = DataFrame(np.ones((100, 10)), columns=['col_' + str(x) for x in range(10)])

    def transform(dataframe):
        dataframe['col_1'] = 10
        dataframe['col_2'] = dataframe['col_0'] + dataframe['col_3']
        return dataframe

    return df.map(transform)


@pytest.fixture
def df_random():
    return DataFrame(np.random.normal(0, 1, (100, 10)), columns=['col_' + str(x) for x in range(10)])


@pytest.fixture
def df_random_with_nans():
    nd = np.random.normal(0, 1, (100, 10))
    sel = np.random.randint(0, 10, 20)
    nd[np.arange(20), sel] = np.nan
    df = DataFrame(nd, columns=['col_' + str(x) for x in range(10)])
    df['categorical'] = ['a'] * 100
    return df


@pytest.fixture
def df_random_unbalanced():
    return DataFrame(np.random.normal(1, 2, (100, 10)), columns=['col_' + str(x) for x in range(10)])


@pytest.fixture
def df_random_large():
    return DataFrame(np.random.normal(0, 1, (1000, 10)), columns=['col_' + str(x) for x in range(10)])


@pytest.fixture
def df_random_datetime():
    df = DataFrame(np.random.normal(0, 1, (1000, 2)), columns=['col_0', 'col_1'])
    nd = (np.random.randint(0, 1582506, (1000, 2)) * 1e12).astype('datetime64[ns]')  # 2020-02-24
    df['dt_1'] = nd[:, 0]
    df['dt_2'] = nd[:, 1]
    df.iloc[:3, -1] = np.nan
    df['dt_3'] = [datetime.datetime(year=1984, month=1, day=1)] * 1000
    df['str'] = ['alpha'] * 1000
    return df.targetize('col_0')


@pytest.fixture
def df_numeric_0():
    return DataFrame({'a': [2, 3, 4] * 3,
                      'b': [1, 5, 6] * 3,
                      'c': [8.0, 7.0, 3.0] * 3})


@pytest.fixture
def df_numeric_1():
    return DataFrame({'a': [1, 2, 3] * 3,
                      'b': [5, 3, 1] * 3,
                      'c': [0.0, 3.0, 7.0] * 3})


@pytest.fixture
def df_datetimeindex():
    return DataFrame({'a': [10, 20, 30, np.nan, 50],
                      'b': [100, 200, np.nan, 400, 500]},
                     index=pd.DatetimeIndex(['2018-02-27 09:01:00',
                                             '2018-02-27 09:02:00',
                                             '2018-02-27 09:03:00',
                                             '2018-02-27 09:04:00',
                                             '2018-02-27 09:05:00']))


@pytest.fixture
def df_missing():
    return DataFrame({'a': [np.nan, 0, 2] * 3,
                      'b': [7, np.nan, 1] * 3,
                      'c': [2, 5, 3] * 3,
                      'd': [1.2, np.nan, 6.7] * 3,
                      'e': [1.6, 1.8, 6.9] * 3,
                      'f': [5, np.nan, np.nan] * 3})


@pytest.fixture
def df_missing_obj():
    return DataFrame({'a': [np.nan, 0, 2] * 3,
                      'b': [7, np.nan, 1] * 3,
                      'c': ["a", "d", "a"] * 3,
                      'd': ["a", np.nan, "b"] * 3,
                      'e': [2, 5, 3] * 3,
                      'f': [2.1, 5.1, 3.1] * 3})


@pytest.fixture
def df_missing_bool():
    return DataFrame({'a': [np.nan, 0, 2] * 3,
                      'b': [7, np.nan, 1] * 3,
                      'c': [True, False, True] * 3,
                      'd': [False, np.nan, True] * 3,
                      'e': [2, 5, 3] * 3,
                      'f': [2.1, 5.1, 3.1] * 3})


@pytest.fixture
def df_missing_blockwise():
    return DataFrame({'a': [1.6, 1.8, 6.9] * 3,
                      'b': [7, np.nan, 1.1] * 3,
                      'c': [2.8, 5.3, 3.7] * 3,
                      'd': [1.2, np.nan, 6.7] * 3,
                      'e': [np.nan, 1.8, 6.9] * 3,
                      'f': [5.0, np.nan, np.nan] * 3})


@pytest.fixture
def df_missing_nan():
    return DataFrame({'a': [1.6, 1.8, 6.9] * 3,
                      'b': [7, np.nan, 1.1] * 3,
                      'c': [2.8, 5.3, 3.7] * 3,
                      'd': [1.2, np.nan, 6.7] * 3,
                      'e': [np.nan, 1.8, 6.9] * 3,
                      'f': [np.nan, np.nan, np.nan] * 3})


@pytest.fixture
def df_bool():
    return DataFrame({'a': [True, True, True] * 3,
                      'b': [False, False, False] * 3,
                      'c': [True, False, True] * 3})


@pytest.fixture
def df_random_categorical():
    # is not completely categorical, so that we can test that too
    df = DataFrame(np.random.normal(0, 1, (100, 2)), columns=['num_col_0', 'num_col_1'])
    df['cat_num_col_0'] = np.random.randint(-3, 3, 100) * math.pi
    df['cat_num_col_1'] = np.random.randint(-10, -2, 100)
    df['cat_num_col_2'] = np.random.randint(-5, 3, 100) * math.e
    df['cat_num_col_0'].iloc[0:5] = np.nan
    df['cat_str_col_0'] = np.random.randint(0, 3, 100)
    df['cat_str_col_0'] = df['cat_str_col_0'].replace({0: 'alpha', 1: 'beta', 2: 'gamma'})
    df['cat_str_col_1'] = df['cat_str_col_0'].replace({'alpha': 'a', 'beta': 'b', 'gamma': 'c'})
    df['cat_dt_col_0'] = df['cat_str_col_1'].replace(
        {'a': datetime.datetime(2012, 12, 20), 'b': datetime.datetime(2015, 1, 1),
         'c': datetime.datetime(2016, 12, 25)})
    df['cat_dt_col_1'] = df['cat_dt_col_0'].replace({'a': np.nan})
    df['cat_str_col_1'].iloc[0] = np.nan

    return df


@pytest.fixture
def operation():
    def multiplier(df):
        for i in range(df.shape[1]):
            df.iloc[:, i] = df.iloc[:, i] * i
        return df

    return multiplier


@pytest.fixture
def csv_path(df_random):
    path = 'tmp.csv'
    df_random.to_csv(path)
    yield path
    os.remove(path)
