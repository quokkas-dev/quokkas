import numpy as np
import pandas as pd
from quokkas.core.frames.dataframe import DataFrame


def test_functions_pipeline():
    def func_to_apply(x):
        return x * 2

    funcs_to_test = {'droplevel': {'level': 'level_2', 'axis': 1},
                     'rename_axis': {'mapper': 'meow', 'axis': 1}, 'abs': {},
                     'filter': {'items': ['a', 'c']}, 'pipe': {'func': func_to_apply}}

    for key, value in funcs_to_test.items():
        if key in ['droplevel']:
            df = DataFrame([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]
            ]).set_index([0, 1])
            df.columns = pd.MultiIndex.from_tuples([
                ('c', 'e'), ('d', 'f')
            ], names=['level_1', 'level_2'])
            df.clear_pipeline()
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
    def func_to_apply(x):
        return x * 2

    funcs_to_test = {'rename_axis': {'mapper': 'meow', 'axis': 1, 'inplace': True}}

    for key, value in funcs_to_test.items():
        if key in ['droplevel']:
            df = DataFrame([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]
            ]).set_index([0, 1])
            df.columns = pd.MultiIndex.from_tuples([
                ('c', 'e'), ('d', 'f')
            ], names=['level_1', 'level_2'])
            df.clear_pipeline()
        else:
            df = DataFrame({'a': [1, 2, 3] * 3,
                            'b': [True, False, True] * 3,
                            'c': [0.0, 3.0, 7.0] * 3})
        method = getattr(df, key)
        new_df = method(**funcs_to_test[key])
        assert new_df is None, 'inplace operation failed'
        assert df.pipeline, 'returned object has no pipeline'
        assert len(df.pipeline._transformations) == 1, 'returned object has more than one transformation'


def test_copy(df_pipelined):
    df_copy = df_pipelined.copy(deep=False)
    df_deepcopy = df_pipelined.copy(deep=True)
    assert df_copy.equals(df_pipelined), 'copy was unsuccessful'
    assert df_deepcopy.equals(df_pipelined), 'copy was unsuccessful'
    assert df_deepcopy.pipeline._transformations, 'pipeline was not copied successfully'
    assert df_deepcopy.pipeline.equals(df_pipelined.pipeline)
    assert df_deepcopy.pipeline is not df_pipelined.pipeline


def test_swapaxes(df_pipelined):
    df_swapped = df_pipelined.swapaxes(0, 1, copy=True)
    assert df_swapped.pipeline, 'pipeline of the swapped object was not initialized'
    assert len(df_swapped.pipeline._transformations) == len(
        df_pipelined.pipeline._transformations) + 1, 'the operation was not pipelined'
    assert df_swapped.pipeline._transformations[-1].func_name == 'swapaxes', 'the name was ' \
            + df_swapped.pipeline._transformations[-1].func_name + ', expected swapaxes'
    df_swapped = df_pipelined.swapaxes(0, 1, copy=False)
    assert df_swapped.pipeline, 'pipeline of the swapped object was not initialized'
    assert len(df_swapped.pipeline._transformations) == len(
        df_pipelined.pipeline._transformations) + 1, 'the operation was not pipelined'
    assert df_swapped.pipeline._transformations[-1].func_name == 'swapaxes', 'the name was ' \
            + df_swapped.pipeline._transformations[-1].func_name + ', expected swapaxes'


def test_column_operations(df):
    df['col_11'] = df['col_1'] + df['col_2']
    assert (df['col_11'] == 2).all(), 'addition of two columns unsuccessful'
    df['col_12'] = df['col_11'] ** 2
    assert (df['col_12'] == 4).all(), 'squaring of a column unsuccessful'
    df['col_13'] = df['col_12'] - 5
    assert (df['col_13'] == -1).all(), 'subtracting a number from a column unsuccessful'
    df.col_0 += df.col_13
    assert (df.col_0 == 0).all(), 'addition of a column to another column is unsuccessful'


def test_align():
    df = DataFrame(
        [[1, 2, 3, 4], [6, 7, 8, 9]], columns=["D", "B", "E", "A"], index=[1, 2]
    )
    other = DataFrame(
        [[10, 20, 30, 40], [60, 70, 80, 90], [600, 700, 800, 900]],
        columns=["A", "B", "C", "D"],
        index=[2, 3, 4],
    )

    def multiplier(df):
        for i in range(df.shape[1]):
            df.iloc[:, i] = df.iloc[:, i] * i
        return df

    left_exp = DataFrame([[12., 2., np.nan, 0., 6.],
                          [27., 7., np.nan, 0., 16.],
                          [np.nan, np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan, np.nan]], index=[1, 2, 3, 4],
                         columns=['A', 'B', 'C', 'D', 'E'])

    right_exp = DataFrame([[np.nan, np.nan, np.nan, np.nan, np.nan],
                           [10., 20., 30., 40., np.nan],
                           [60., 70., 80., 90., np.nan],
                           [600., 700., 800., 900., np.nan]], index=[1, 2, 3, 4],
                          columns=['A', 'B', 'C', 'D', 'E'])

    df.map(multiplier)
    left, right = df.align(other)
    assert left.equals(left_exp)
    assert right.equals(right_exp)
    assert left.pipeline
    assert len(left.pipeline._transformations) == 1
    assert not right.pipeline._transformations


def test_functions_preserved(df_numeric_0, df_numeric_1):
    funcs_to_test = {'take': {'indices': [0, 1]}, 'get': {'key': ['a', 'b']},
                     'head': {'n': 2}, 'tail': {'n': 2}, 'sample': {}}

    for key, value in funcs_to_test.items():
        df = df_numeric_0.copy(deep=True)
        method = getattr(df, key)
        new_df = method(**funcs_to_test[key])
        assert df.pipeline, 'returned object has no pipeline'
        assert df.pipeline._transformations is None, 'returned object has transformations'
        assert new_df.pipeline, 'returned object has no pipeline'
        assert new_df.pipeline._transformations is None, 'returned object has more than one transformation'
