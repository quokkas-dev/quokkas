import pandas as pd, numpy as np
from quokkas.core.frames.dataframe import DataFrame
from quokkas.utils.test_utils import approximately_equal


def test_create_df(df):
    assert df.shape == (10000, 10), 'shape is wrong'
    assert isinstance(df, DataFrame), 'the created object is not an instance of pandas.DataFrame'
    assert isinstance(df.iloc[:, 9], pd.Series), 'the column of the created object is not an instance of pd.Series'
    assert df.iloc[-1, -1] == 1, 'the last value in the created object is != 1'


def test_transform(df):
    def transform(dataframe):
        dataframe['col_1'] = 0
        dataframe['col_2'] = dataframe['col_0'] + dataframe['col_3']
        return dataframe

    def transform_abs(dataframe):
        return dataframe.abs()

    df.transform(transform)
    assert isinstance(df, DataFrame), 'returned object is not a DataFrame'
    assert (df['col_1'] == 0).all(), 'transformation was unsuccessful'
    new_df = df.transform(transform_abs)
    assert isinstance(new_df, DataFrame), 'returned object is not a DataFrame'
    assert df.pipeline, 'returned object has no pipeline'
    assert len(df.pipeline._transformations) == 1
    assert len(new_df.pipeline._transformations) == 2


def test_T_pipeline(df):
    df = df.T
    assert df.pipeline._transformations
    assert len(df.pipeline._transformations) == 1


def test_resample_pipeline(df):
    pass


def test_functions_pipeline():
    def func_to_apply(x):
        return x * 2

    funcs_to_test = {'transpose': {}, 'query': {'expr': 'a > b'},
                     'select_dtypes': {'include': ['int64']}, 'drop': {'columns': ['a', 'b']},
                     'rename': {'mapper': {'a': 'e', 'b': 'f'}},
                     'fillna': {'value': 0},
                     'bfill': {},
                     'ffill': {},
                     'drop_duplicates': {'subset': ['a', 'b']},
                     'dropna': {'axis': 1},
                     'sort_values': {'by': 'a'},
                     'sort_index': {'axis': 0},
                     'reset_index': {},
                     'replace': {'to_replace': [1, 2, 3], 'value': 0},
                     'astype': {'dtype': 'int32'},
                     'apply': {'func': func_to_apply},
                     'applymap': {'func': func_to_apply},
                     'round': {'decimals': 1}, 'corr': {}, 'cov': {}, 'asfreq': {'freq': '30S'},
                     'interpolate': {'method': 'linear', 'limit_direction': 'forward', 'axis': 0},
                     'diff': {}, 'shift': {}, 'stack': {}, 'unstack': {}, 'explode': {'column': 'A'},
                     'melt': {'id_vars': ['a'], 'value_vars': ['b']}, 'to_timestamp': {'copy': False},
                     'to_period': {'copy': False},
                     'clip': {'lower': 0, 'upper': 10},
                     'reorder_levels': {'order': ["diet", "class"]},
                     'swaplevel': {'i': 0, 'j': 1}
                     }

    for key, value in funcs_to_test.items():
        if key in ['fillna', 'bfill', 'ffill', 'asfreq', 'interpolate', 'diff', 'shift', 'to_period']:
            df = DataFrame({'a': [10, 20, 30, np.nan, 50],
                            'b': [100, 200, np.nan, 400, 500]},
                           index=pd.DatetimeIndex(['2018-02-27 09:01:00',
                                                   '2018-02-27 09:02:00',
                                                   '2018-02-27 09:03:00',
                                                   '2018-02-27 09:04:00',
                                                   '2018-02-27 09:05:00']))
        elif key in ['reorder_levels', 'swaplevel']:
            df = DataFrame({"class": ["Mammals", "Mammals", "Reptiles"],
                    "diet": ["Omnivore", "Carnivore", "Carnivore"],
                    "species": ["Humans", "Dogs", "Snakes"]},
                    columns=["class", "diet", "species"]).set_index(["class", "diet"])
            df.clear_pipeline()
        elif key in ['to_timestamp']:
            df = DataFrame(["DBMS", "DSA", "OOPS", "System Design", "CN", ],
                            index=pd.period_range('2020-08-15', periods=5), columns=['Course'])
        elif key in ['stack']:
            multicol2 = pd.MultiIndex.from_tuples([('weight', 'kg'),
                                                   ('height', 'm')])
            df = DataFrame([[1.0, 2.0], [3.0, 4.0]],
                                                index=['cat', 'dog'],
                                                columns=multicol2)
        elif key in ['unstack']:
            index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),
                                               ('two', 'a'), ('two', 'b')])
            df = DataFrame([[1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [1.0, 2.0]],
                              index=index,
                              columns=['cat', 'dog'])
        elif key in ['explode']:
            df = DataFrame({'A': [[0, 1, 2], 'foo', [], [3, 4]],
                               'B': 1,
                               'C': [['a', 'b', 'c'], np.nan, [], ['d', 'e']]})
        else:
            df = DataFrame({'a': [1, 2, 3] * 3,
                            'b': [True, False, True] * 3,
                            'c': [0.0, 3.0, 7.0] * 3})

        method = getattr(df, key)
        new_df = method(**funcs_to_test[key])
        assert df.pipeline, 'returned object has no pipeline'
        assert df.pipeline._transformations is None, 'returned object has transformations'
        assert new_df.pipeline, 'returned object has no pipeline'
        assert len(new_df.pipeline._transformations) == 1, 'returned object has more than one transformation'


def test_functions_inplace_pipeline():
    funcs_to_test = {'query': {'expr': 'a > b', 'inplace': True},
                     'drop': {'columns': ['a', 'b'], 'inplace': True},
                     'rename': {'mapper': {'a': 'e', 'b': 'f'}, 'inplace': True},
                     'fillna': {'value': 0, 'inplace': True},
                     'bfill': {'inplace': True},
                     'ffill': {'inplace': True},
                     'drop_duplicates': {'subset': ['a', 'b'], 'inplace': True},
                     'dropna': {'axis': 1, 'inplace': True},
                     'sort_values': {'by': 'a', 'inplace': True},
                     'sort_index': {'axis': 0, 'inplace': True},
                     'interpolate': {'method': 'linear', 'limit_direction': 'forward', 'axis': 0,
                                     'inplace': True},
                     'replace': {'to_replace': [1, 2, 3], 'value': 0, 'inplace': True},
                     'clip': {'lower': 0, 'upper': 10, 'inplace': True}}

    for key, value in funcs_to_test.items():
        if key in ['fillna', 'bfill', 'ffill', 'asfreq', 'interpolate', 'diff', 'shift', 'to_period']:
            df = DataFrame({'a': [10, 20, 30, np.nan, 50],
                            'b': [100, 200, np.nan, 400, 500]},
                           index=pd.DatetimeIndex(['2018-02-27 09:01:00',
                                                   '2018-02-27 09:02:00',
                                                   '2018-02-27 09:03:00',
                                                   '2018-02-27 09:04:00',
                                                   '2018-02-27 09:05:00']))
        elif key in ['to_timestamp']:
            df = DataFrame(["DBMS", "DSA", "OOPS", "System Design", "CN", ],
                            index=pd.period_range('2020-08-15', periods=5), columns=['Course'])
        else:
            df = DataFrame({'a': [1, 2, 3] * 3,
                            'b': [True, False, True] * 3,
                            'c': [0.0, 3.0, 7.0] * 3})
        method = getattr(df, key)
        method(**funcs_to_test[key])
        assert df.pipeline, 'returned object has no pipeline'
        assert len(df.pipeline._transformations) == 1, 'returned object has more than one transformation'


def test_functions_preserved(df_numeric_0, df_numeric_1, df_datetimeindex, df_bool):
    funcs_to_test = {'where': {'cond': df_bool, 'other': df_numeric_1},
                     'mask': {'cond': df_bool, 'other': df_numeric_1},
                    'dot': {'other': df_datetimeindex.T}}

    for key, value in funcs_to_test.items():
        if key in ['dot']:
            df = DataFrame({'a': [10, 20, 30, np.nan, 50],
                            'b': [100, 200, np.nan, 400, 500]},
                           index=pd.DatetimeIndex(['2018-02-27 09:01:00',
                                                   '2018-02-27 09:02:00',
                                                   '2018-02-27 09:03:00',
                                                   '2018-02-27 09:04:00',
                                                   '2018-02-27 09:05:00']))
        elif key in ['to_timestamp']:
            df = DataFrame(["DBMS", "DSA", "OOPS", "System Design", "CN", ],
                           index=pd.period_range('2020-08-15', periods=5), columns=['Course'])
        else:
            df = df_numeric_0
        method = getattr(df, key)
        new_df = method(**funcs_to_test[key])
        assert df.pipeline, 'returned object has no pipeline'
        assert df.pipeline._transformations is None, 'returned object has transformations'
        assert new_df.pipeline, 'returned object has no pipeline'
        assert new_df.pipeline._transformations is None, 'returned object has more than one transformation'


def test_functions_inplace_preserved(df_numeric_1, df_datetimeindex, df_bool):

    funcs_to_test = {'where': {'cond': df_bool, 'other': df_numeric_1, 'inplace': True},
                     'mask': {'cond': df_bool, 'other': df_numeric_1, 'inplace': True}}

    for key, value in funcs_to_test.items():
        if key in ['fillna', 'bfill', 'ffill', 'asfreq', 'interpolate', 'diff', 'shift', 'to_period']:
            df = DataFrame({'a': [10, 20, 30, np.nan, 50],
                            'b': [100, 200, np.nan, 400, 500]},
                           index=pd.DatetimeIndex(['2018-02-27 09:01:00',
                                                   '2018-02-27 09:02:00',
                                                   '2018-02-27 09:03:00',
                                                   '2018-02-27 09:04:00',
                                                   '2018-02-27 09:05:00']))
        elif key in ['to_timestamp']:
            df = DataFrame(["DBMS", "DSA", "OOPS", "System Design", "CN", ],
                           index=pd.period_range('2020-08-15', periods=5), columns=['Course'])
        else:
            df = DataFrame({'a': [1, 2, 3] * 3,
                            'b': [5, 3, 1] * 3,
                            'c': [0.0, 3.0, 7.0] * 3})
        method = getattr(df, key)
        new_df = method(**funcs_to_test[key])
        assert new_df is None, 'inplace operation failed'
        assert df.pipeline, 'returned object has no pipeline'
        assert df.pipeline._transformations is None, 'returned object has more than one transformation'


def test_getitem_pipeline(df_numeric_1):
    df_numeric_1.drop(['c'], axis=1, inplace=True)
    new_df = df_numeric_1[['a', 'b']]
    assert new_df.pipeline
    assert df_numeric_1.pipeline, 'returned object has no pipeline'
    assert len(df_numeric_1.pipeline._transformations) == 1, 'returned object has more than one transformation'
    assert len(new_df.pipeline._transformations) == 1, 'returned object has more than one transformation'


def test_align_pipeline(df_numeric_0, df_numeric_1):
    df_numeric_0.drop(['c'], axis=1, inplace=True)
    df_numeric_1.drop(['c'], axis=1, inplace=True)
    df_1 = df_numeric_0.align(df_numeric_1)
    assert len(df_numeric_1.pipeline._transformations) == 1, 'returned object has more than one transformation'
    assert len(df_numeric_0.pipeline._transformations) == 1, 'returned object has more than one transformation'
    assert len(df_1[0].pipeline._transformations) == 1, 'returned object has more than one transformation'
    assert len(df_1[0].pipeline._transformations) == 1, 'returned object has more than one transformation'


def test_compare_pipeline(df_numeric_0):
    df2 = df_numeric_0.copy()
    df2.loc[0, 'a'] = 'c'
    df2.loc[2, 'b'] = 4.0
    df_numeric_0.drop(['c'], axis=1, inplace=True)
    df2.drop(['c'], axis=1, inplace=True)
    new_df = df_numeric_0.compare(df2)
    assert df2.pipeline
    assert new_df.pipeline
    assert df_numeric_0.pipeline
    assert len(df_numeric_0.pipeline._transformations) == 1
    assert len(new_df.pipeline._transformations) == 1
    assert len(df_numeric_0.pipeline._transformations) == 1


def test_double_functions_pipeline(df_numeric_0, df_numeric_1):
    def func_to_apply(x1, x2):
        return x1 + x2
    df_numeric_0.drop(['c'], axis=1, inplace=True)
    df_numeric_1.drop(['c'], axis=1, inplace=True)
    funcs_to_test = {'combine': {'other': df_numeric_1.copy(deep=True), 'func': func_to_apply},
                     'combine_first': {'other': df_numeric_1.copy(deep=True)},
                     'join': {'other': df_numeric_1.copy(deep=True), 'how': 'outer', 'lsuffix': '_left',
                              'rsuffix': '_right'},
                     'append': {'other': df_numeric_1.copy(deep=True)},
                     'concat': {'others': [df_numeric_1.copy(deep=True), df_numeric_1.copy(deep=True)]},
                     'update': {'other': df_numeric_1.copy(deep=True)},
                     'merge': {'other': df_numeric_1.copy(deep=True), 'on': 'a', 'how': 'outer'},
                     }

    for key, value in funcs_to_test.items():
        df = df_numeric_0.copy(deep=True)
        df1 = df_numeric_1.copy(deep=True)
        if key in ['merge']:
            new_df = df.merge(df1, on='a', how='outer')

        elif key in ['concat']:
            if key == "concat":
                new_df = pd.concat([df, df1], axis=0)
            assert len(df.pipeline._transformations) == 1
            assert len(df1.pipeline._transformations) == 1
            assert new_df.pipeline._transformations is None
        elif key in ['update']:
            new_df = df.update(df1)
            assert len(df.pipeline._transformations) == 1
            assert len(df1.pipeline._transformations) == 1
            assert new_df is None
        else:
            method = getattr(df, key)
            new_df = method(**funcs_to_test[key])
            assert df.pipeline
            assert df1.pipeline
            assert new_df.pipeline
            assert len(df.pipeline._transformations) == 1
            assert len(df1.pipeline._transformations) == 1
            assert len(new_df.pipeline._transformations) == 1


def test_pop_pipeline(df):
    df_copy = df.copy(deep=True)
    series = df.pop('col_0')
    assert series.equals(df_copy.col_0)
    df_copy.stream(df)
    assert 'col_0' not in df_copy.columns
    assert 'col_1' in df_copy.columns
    assert 'col_0' not in df.columns
    assert len(df.pipeline._transformations) == 1, 'pop operation failed'
    assert len(df_copy.pipeline._transformations) == 1, 'stream pop operation failed'


def test_asof_pipeline(df, operation):
    df = DataFrame({'a': [10, 20, 30, 40, 50],
                    'b': [100, 200, 300, 400, 500]},
                   index=pd.DatetimeIndex(['2018-02-27 09:01:00',
                                           '2018-02-27 09:02:00',
                                           '2018-02-27 09:03:00',
                                           '2018-02-27 09:04:00',
                                           '2018-02-27 09:05:00']), dtype=int)
    df.map(operation)
    tmp = df.asof(pd.DatetimeIndex(['2018-02-27 09:03:30',
                                    '2018-02-27 09:04:30'])).astype(int)
    assert (tmp.values == [[0, 300], [0, 400]]).all()
    assert (tmp.index == pd.DatetimeIndex(['2018-02-27 09:03:30', '2018-02-27 09:04:30'])).all()

    assert len(tmp.pipeline._transformations) == 2
    assert len(df.pipeline._transformations) == 1


def test_arithmetic_operations_pipeline(df_pipelined, df):
    df_pipelined += df
    assert len(df_pipelined.pipeline._transformations) == 1
    assert not df.pipeline._transformations


def test_loc_iloc(df_pipelined, operation):
    iloced = df_pipelined.iloc[:50, :2]
    loced = df_pipelined.loc[0:2]
    assert iloced.shape == (50, 2)
    assert len(iloced.pipeline._transformations) == 1
    assert loced.shape == (3, df_pipelined.shape[1])
    assert len(loced.pipeline._transformations) == 1
    df_pipelined.map(operation)
    assert len(iloced.pipeline._transformations) == 1
    assert len(loced.pipeline._transformations) == 1
    assert len(df_pipelined.pipeline._transformations) == 2


def test_to_from_pandas():
    pdf = pd.DataFrame(np.random.normal(0, 1, (100, 10)), columns=['col_' + str(i) for i in range(10)])
    df = DataFrame.from_pandas(pdf, deep=False)
    df_deep = DataFrame.from_pandas(pdf, deep=True)
    assert isinstance(df, DataFrame)
    zero_value = pdf['col_0'][0]
    df['col_0'][0] = 100
    assert pdf['col_0'][0] == 100
    assert df_deep['col_0'][0] == zero_value
    new_pdf = df.to_pandas()
    assert isinstance(new_pdf, pd.DataFrame)
    assert (new_pdf.columns == pdf.columns).all()
    assert new_pdf['col_0'][0] == 100


def test_targetize(df_numeric_0):
    df = df_numeric_0
    df.targetize(['a'])
    assert df.target == {'a'}

    df.drop(['a'], axis=1, inplace=True)
    assert df.target == {'a'}

    df.targetize('c')
    assert df.target == {'c'}

    df_numeric_0.targetize(['a', 'b'])
    assert df_numeric_0.target == {'a', 'b'}


def test_datetime(df_random_datetime):
    df = df_random_datetime.astype('object')
    transformed = df.to_datetime(['dt_1', 'dt_2', 'dt_3'], inplace=False)
    for i in range(3):
        assert transformed['dt_' + str(i + 1)].dtype.kind == 'M'
    assert transformed.pipeline._transformations[-1].func_name == 'to_datetime'

    transformed['year'] = transformed['dt_1'].dt.year
    transformed['month'] = transformed['dt_1'].dt.month
    transformed['day'] = transformed['dt_1'].dt.day

    transformed.to_datetime(['year', 'month', 'day'], inplace=True, column_names='dt_4')
    assert 'year' not in transformed.columns
    assert (transformed['dt_4'].dt.day == transformed['dt_1'].dt.day).all()


def test_separate(df_random):
    (n, m) = df_random.shape
    X, y = df_random.separate(to_numpy=True)

    assert y is None
    assert isinstance(X, np.ndarray)
    assert X.shape == (n, m)

    df_random['target'] = df_random['col_0'] * df_random['col_1']
    X, y = df_random.separate(to_numpy=False, target='target', squeeze=True)
    assert y.shape == (n,)
    assert X.shape == (n, m)
    assert df_random.target == {'target'}

    X, y = df_random.separate(to_numpy=True, squeeze=True)
    assert isinstance(y, np.ndarray) and y.shape == (n, )
    assert isinstance(X, np.ndarray) and X.shape == (n, m)

    X, y = df_random.separate(inplace=True, squeeze=False)
    assert y.shape == (n, 1)
    assert X.shape == (n, m)
    assert df_random.shape == (n, m)


def test_capply(df_random):
    m = df_random.shape[1]

    df_random.capply(lambda x: x * x, 'col_0', 'transformed', inplace=True)
    assert df_random.shape[1] == m + 1
    assert approximately_equal(df_random['transformed'], df_random['col_0'] ** 2)

    df_random.capply(lambda x: np.sqrt(x), 'transformed', inplace=True)
    assert approximately_equal(df_random['transformed'], df_random['col_0'].abs())

    df_random.capply(lambda x: pd.Series([x > 1, x < 0.5]), 'transformed', inplace=True)
    assert df_random.shape[1] == m + 3
    assert (df_random[0] == (df_random['transformed'] > 1)).all()

    df = df_random.capply(lambda x: pd.Series([x > 0, x < 0]), 'col_1', ['positive', 'negative'])
    assert df.shape[1] == m + 5
    assert df_random.shape[1] == m + 3
    assert (df['positive'] == (df_random['col_1'] > 0)).all()

    assert len(df.pipeline._transformations) == 4
    assert df.pipeline._transformations[-1].func_name == "capply"
