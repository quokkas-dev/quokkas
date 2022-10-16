import numbers

import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from pandas import unique
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import KNNImputer as SklearnKNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer as SklearnSimpleImputer, IterativeImputer as SklearnIterativeImputer
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from quokkas.core.frames.dataframe import DataFrame
from quokkas.transforms.date_encoder import DateEncoder
from quokkas.transforms.encoders import OneHotEncoder
from quokkas.transforms.imputers import SimpleImputer, IterativeImputer, KNNImputer
from quokkas.transforms.scalers import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from quokkas.transforms.splitters import _BaseSplitter
from quokkas.utils.sk_utils import is_scalar_nan
from quokkas.utils.test_utils import approximately_equal, isnan_safe_1d, series_is_in, approximately_equal_scalars
from quokkas.transforms.encoders import OrdinalEncoder as qkOrdinalEncoder


def test_streamer_base(df):
    def multiplier(df):
        for i in range(df.shape[1]):
            df.iloc[:, i] = df.iloc[:, i] * i
        return df

    def not_inplace_adder(df):
        df = df.copy(deep=True)
        df['col_0'] = df['col_0'] + df['col_9']
        return df

    df.map(multiplier)
    assert (df['col_9'] == 9).all(), 'the transform was unsuccessful'
    assert df.pipeline._transformations[0].func == multiplier

    new_df = df.map(not_inplace_adder)
    assert (df['col_0'] == 0).all(), 'the transform was completed inplace'
    assert len(df.pipeline._transformations) == 1
    assert (new_df['col_0'] == 9).all(), 'the transform was unsuccessful'
    assert len(new_df.pipeline._transformations) == 2


def test_streamer_pipeline(df):
    df_zeros = np.zeros_like(df)
    df_ones = np.ones_like(df)

    def itemwise_multiplier(dataframe, mul):
        return dataframe * mul

    new_df = df.map(itemwise_multiplier, df_ones)
    assert (new_df == df).all().all()
    assert new_df.pipeline._transformations
    assert len(new_df.pipeline._transformations) == 1

    parallel_df = new_df.pipeline.fit_transform(df, parameters=df_zeros)
    assert (parallel_df == 0).all().all()
    assert parallel_df.pipeline._transformations
    assert len(parallel_df.pipeline._transformations) == 1

    random_df = DataFrame(np.random.normal(0, 1, df.shape))
    random_df = random_df.abs()
    parallel_df = new_df.pipeline.fit_transform(random_df)
    assert (parallel_df == random_df).all().all()
    assert parallel_df.pipeline._transformations
    assert len(parallel_df.pipeline._transformations) == 2


def test_external(df_random_unbalanced):
    transformed = df_random_unbalanced.external(SklearnStandardScaler())
    assert transformed.pipeline._transformations is not None
    assert df_random_unbalanced.pipeline._transformations is None
    assert isinstance(transformed.pipeline._transformations[0].processor, SklearnStandardScaler)
    assert ((transformed.std() - 1).abs() < 1e-2).all()
    assert ((transformed.mean()).abs() < 1e-9).all()

    retransformed = df_random_unbalanced.stream(transformed)
    assert retransformed.equals(transformed)

    class InplaceScaler:
        def __init__(self):
            self.std = None

        def fit(self, df):
            self.std = np.array(df.std()).squeeze()

        def transform(self, df):
            for i, col in enumerate(df.columns):
                df[col] /= self.std[i]

    df_random_unbalanced.external(InplaceScaler(), inplace=True)
    assert ((df_random_unbalanced.std() - 1).abs() < 1e-2).all()


def test_onehot_encoder(df_random_categorical):
    cat_columns = df_random_categorical.columns[2:]
    transformed = df_random_categorical.encode(kind='onehot', inplace=False, dtype=np.float64, auto=True,
                                               keep_original=True, handle_unknown='ignore')
    _check_encoding(df_random_categorical, transformed, cat_columns)

    transformed_parallel = df_random_categorical.encode(kind='onehot', inplace=False, dtype=np.float64, auto=True,
                                                        keep_original=False, handle_unknown='ignore')
    assert transformed_parallel.pipeline._encoded_cols == (set(transformed_parallel.columns) - set(df_random_categorical.columns))

    transformed.drop(cat_columns, axis=1, inplace=True)
    assert approximately_equal(transformed, transformed_parallel)


    transformed_parallel = df_random_categorical.encode(kind='onehot', inplace=False, dtype=np.float64,
                                                        include=cat_columns,
                                                        auto=False, keep_original=False, handle_unknown='ignore')
    assert approximately_equal(transformed_parallel, transformed)

    transformed_parallel = df_random_categorical.encode(kind='onehot', inplace=False, dtype=np.int32,
                                                        include=cat_columns,
                                                        auto=False, keep_original=False, sparse=True,
                                                        handle_unknown='ignore')
    for i in set(transformed_parallel.columns) - set(df_random_categorical.columns):
        transformed_parallel[i] = transformed_parallel[i].sparse.to_dense()

    assert approximately_equal(transformed, transformed_parallel)

    transformed_parallel = df_random_categorical.stream(transformed)
    assert approximately_equal(transformed_parallel, transformed)

    df_new = df_random_categorical.copy(deep=True)
    mod_cols = [cat_columns[3], cat_columns[0]]
    df_new[cat_columns[3]].iloc[0] = 'alphabeta'
    df_new[cat_columns[0]].iloc[0] = 99

    transformed_parallel = df_new.stream(transformed)
    assert transformed.pipeline.equals(transformed_parallel.pipeline)

    for col in mod_cols:
        uniques = unique(df_random_categorical[col])
        for u in uniques:
            assert transformed_parallel[col + '_' + str(u)].iloc[0] == 0

    df_random_categorical.encode(kind='onehot', inplace=True, dtype=np.int32,
                                 include=cat_columns,
                                 auto=False, keep_original=False, sparse=True,
                                 handle_unknown='ignore')
    assert approximately_equal(df_random_categorical, transformed)

    transformed = df_random_categorical.encode(kind='onehot', inplace=False, dtype=np.int32,
                                               auto=True)

    assert df_random_categorical.equals(transformed)


def test_onehot_encoder_infrequent(df_random_categorical):
    cat_columns = df_random_categorical.columns[2:]
    # let's add a couple of sparse values
    df_random_categorical['cat_num_col_0'].iloc[
        50] = 17.0  # pandas doesn't like such assignments and throws lots of warnings
    df_random_categorical['cat_num_col_0'].iloc[51] = 18.0

    df_random_categorical['cat_str_col_0'].iloc[50] = 'delta'
    df_random_categorical['cat_str_col_0'].iloc[51] = 'mu'

    # some config
    min_count = 2
    min_percentage = (2 / df_random_categorical.shape[0]) - 1e-8

    transformed = df_random_categorical.encode(kind='onehot', inplace=False, dtype=np.float64, auto=True,
                                               keep_original=True, handle_unknown='ignore', min_frequency=min_count)
    _check_encoding(df_random_categorical, transformed, cat_columns, infrequent_border=min_count)
    transformed.drop(cat_columns, axis=1, inplace=True)

    transformed_parallel = df_random_categorical.encode(kind='onehot', inplace=False, dtype=np.float64, auto=True,
                                                        keep_original=False, handle_unknown='ignore',
                                                        min_frequency=min_percentage)

    assert approximately_equal(transformed, transformed_parallel)

    transformed_parallel = df_random_categorical.encode(kind='onehot', inplace=False, dtype=np.float64, auto=True,
                                                        keep_original=False, handle_unknown='infrequent_if_exist',
                                                        min_frequency=min_percentage)
    assert approximately_equal(transformed, transformed_parallel)

    df_new = df_random_categorical.copy(deep=True)
    df_new['cat_num_col_0'].iloc[50] = 19.0
    transformed_parallel = df_new.stream(transformed_parallel)
    _check_encoding(df_new, transformed_parallel, cat_columns, infrequent_border=min_count)
    assert transformed_parallel['cat_num_col_0_infrequent'].iloc[50] == 1

    transformed_new_drop = df_new.stream(transformed)
    assert transformed_new_drop['cat_num_col_0_infrequent'].iloc[50] == 0

    transformed = df_random_categorical.encode(kind='onehot', inplace=False, dtype=np.int32, auto=True,
                                               keep_original=False, handle_unknown='infrequent_if_exist',
                                               min_frequency=min_percentage)
    assert approximately_equal(transformed, transformed_parallel)

    transformed = df_random_categorical.encode(kind='onehot', inplace=False, dtype=np.int32, auto=True,
                                               keep_original=False, handle_unknown='infrequent_if_exist',
                                               min_frequency=min_percentage, drop='if_binary')
    assert approximately_equal(transformed, transformed_parallel)

    transformed = df_random_categorical.encode(kind='onehot', inplace=False, dtype=np.int32, auto=True,
                                               keep_original=False, handle_unknown='infrequent_if_exist',
                                               min_frequency=min_percentage, drop='first')

    _check_encoding(df_random_categorical, transformed, cat_columns, infrequent_border=2, drop_first=True)

    df_random_categorical.encode(kind='onehot', inplace=True, dtype=np.int32, auto=True,
                                 keep_original=False, handle_unknown='infrequent_if_exist',
                                 min_frequency=min_percentage, drop='first')
    assert approximately_equal(df_random_categorical, transformed)

    transformed = df_random_categorical.encode(kind='onehot', inplace=False, auto=True, keep_original=True)
    assert transformed.equals(df_random_categorical)


def _check_encoding(df, transformed, cols, infrequent_border=None, drop_first=False):
    encoder = transformed.pipeline._transformations[-1]
    assert isinstance(encoder, OneHotEncoder)
    for col in cols:
        uniques, counts = OneHotEncoder._determine_unique(df[col].to_numpy(), return_counts=True)
        for u, count in zip(uniques, counts):
            if drop_first and (u == _find_first_param(encoder, col)):
                assert (col + '_' + str(u)) not in transformed.columns
                continue
            if infrequent_border is not None and count < infrequent_border:
                if not isinstance(u, numbers.Real) or not np.isnan(u):
                    assert series_is_in((df[col] == u), transformed[col + '_infrequent'])
                else:
                    assert series_is_in((isnan_safe_1d(df[col])), transformed[col + '_infrequent'])
            else:
                if not isinstance(u, numbers.Real) or not np.isnan(u):
                    assert (df[col] == u).equals((transformed[col + '_' + str(u)]).astype(np.bool))
                else:
                    assert (isnan_safe_1d(df[col])).equals(
                        (transformed[col + '_' + str(u)]).astype(np.bool))


def _find_first_param(encoder, col):
    if encoder.infrequents is not None and col in encoder.infrequents:
        for p in encoder.params[col]:
            if p not in encoder.infrequents[col] and (not is_scalar_nan(p) or not pd.isnull(encoder.infrequents[col])):
                return p
    else:
        return encoder.params[col][0]


def test_ordinal_encoder(df_random_categorical):
    cat_columns = df_random_categorical.columns[2:]
    df_parallel = df_random_categorical.copy(deep=True)
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder(dtype=np.float64)
    for column in cat_columns:
        df_parallel[column] = oe.fit_transform(df_random_categorical[column].to_numpy().reshape(-1, 1))

    transformed = df_random_categorical.encode(kind='ordinal', inplace=False, dtype=np.float64, auto=True)
    assert approximately_equal(df_parallel, transformed, skipna=True)
    assert len(transformed.pipeline._transformations) == 1
    assert isinstance(transformed.pipeline._transformations[0], qkOrdinalEncoder)

    transformed_parallel = df_random_categorical.encode(kind='ordinal', inplace=False, dtype=np.float64, auto=True,
                                                        handle_unknown='use_encoded_value', unknown_value=100)
    assert transformed_parallel.equals(transformed)

    df_new = df_random_categorical.copy(deep=True)

    df_new[cat_columns[3]].iloc[0] = 'alphabeta'
    df_new[cat_columns[0]].iloc[0] = 99
    df_new = df_new.stream(transformed_parallel)
    assert df_new[cat_columns[3]].iloc[0] == 100
    assert df_new[cat_columns[0]].iloc[0] == 100

    transformed_parallel = df_random_categorical.encode(kind='ordinal', inplace=False, dtype=np.float64, auto=True,
                                                        encoded_missing_value=np.nan)
    assert transformed_parallel.equals(df_parallel)

    transformed_parallel = df_random_categorical.encode(kind='ordinal', inplace=False, dtype=np.int32,
                                                        include=cat_columns)
    assert transformed.equals(transformed_parallel.astype(np.float64))
    assert transformed.pipeline._encoded_cols == set(cat_columns)

    categories = transformed_parallel.pipeline._transformations[-1].params
    categories.pop(df_random_categorical.columns[2])
    transformed_parallel = df_random_categorical.encode(kind='ordinal', inplace=False, dtype=np.int32,
                                                        include=cat_columns, categories=categories)
    assert transformed.equals(transformed_parallel.astype(np.float64))

    df_random_categorical.encode(kind='ordinal', inplace=True, dtype=np.int32,
                                 exclude=df_random_categorical.columns[:2])
    assert transformed_parallel.equals(df_random_categorical)

    transformed = df_random_categorical.encode(kind='ordinal', inplace=False,
                                 exclude=df_random_categorical.columns[:2], auto=True)
    assert transformed.equals(df_random_categorical)


def test_date_encoder(df_random_datetime):
    date_cols = ['dt_1', 'dt_2', 'dt_3']
    transformed = df_random_datetime.encode_dates(inplace=False, auto=True, ordinal=True, intrayear=True,
                                                  intraweek=True, keep_original=False)

    _check_date_encoding(df_random_datetime, transformed, date_cols)

    transformed = df_random_datetime.encode_dates(inplace=False, auto=True, ordinal=True, intraweek=True,
                                                  intrayear=False, keep_original=True)

    _check_date_encoding(df_random_datetime, transformed, date_cols, check_intrayear=False, dropped=False)

    reduced_cols = ['dt_2', 'dt_3']
    transformed = df_random_datetime.encode_dates(include=reduced_cols, ordinal=True, intraweek=False, intrayear=False,
                                                  keep_original=False, inplace=False)
    _check_date_encoding(df_random_datetime, transformed, reduced_cols, check_ordinal=True, check_intraweek=False,
                         check_intrayear=False, dropped=True)
    assert (transformed['dt_1'] == df_random_datetime['dt_1']).all()

    copied = df_random_datetime.copy(deep=True)
    copied.encode_dates(include=reduced_cols, ordinal=True, intraweek=True, intrayear=True, keep_original=True,
                        inplace=True, auto=False)

    assert len(copied.pipeline._transformations) == 2
    _check_date_encoding(df_random_datetime, copied, reduced_cols, check_intraweek=True,
                         check_intrayear=True, check_ordinal=True, dropped=False)

    copied = transformed.encode_dates(exclude=['dt_1'], auto=True)
    assert copied.equals(transformed)


def _check_date_encoding(df, transformed, date_cols, check_ordinal=True, check_intrayear=True,
                         check_intraweek=True, dropped=True):
    assert transformed.target == df.target
    assert transformed.pipeline._transformations[-1].__class__.__name__ == 'DateEncoder'
    for col in date_cols:
        col_name = col + '_ordinal'
        if check_ordinal:
            ordinal = df[col].astype(np.int64)
            amin = ordinal.min()
            amax = ordinal.max()

            if amax == amin:
                assert (transformed[col_name] == 0).all()
            else:
                assert approximately_equal(transformed[col_name], (ordinal - amin) / (amax - amin))

            assert col_name in transformed.pipeline._encoded_cols
        else:
            assert col_name not in transformed.columns

        col_name = col + '_intrayear'
        if check_intrayear:
            years = df[col].to_numpy().astype('datetime64[Y]')
            years_series = pd.Series(years, index=df.index)
            intrayear = (df[col] - years_series).astype(np.int64) / (
                        DateEncoder.YEAR_LENGTH + DateEncoder.DAY_LENGTH * (years.astype(np.int32) % 4 == 0))
            assert approximately_equal(transformed[col_name], intrayear)
            assert col_name in transformed.pipeline._encoded_cols
        else:
            assert col_name not in df.columns

        col_name = col + '_intraweek'
        if check_intraweek:
            assert approximately_equal(transformed[col_name], df[col].dt.dayofweek, skipna=True)
            assert col_name in transformed.pipeline._encoded_cols
        else:
            assert col_name not in df.columns

        if dropped:
            assert col not in transformed.columns
        else:
            assert ((transformed[col] == df[col]) | (transformed[col].isna())).all()


def test_trimmer(df_random_with_nans):
    low = df_random_with_nans.quantile(0.01, axis=0)
    high = df_random_with_nans.quantile(0.99, axis=0)

    transformed = df_random_with_nans.trim(limits=(0.01, 0.01), inplace=False, inclusive=(True, True))
    assert transformed.shape[0] < df_random_with_nans.shape[0]
    _assert_trim_and_compare(transformed, low, high)

    df_random_with_nans.iloc[0, :-1] = 0
    df_random_with_nans.iloc[1, :-1] = -1

    transformed = df_random_with_nans.trim(limits=(-1, 10), inplace=False, relative=False, inclusive=(False, False))
    _assert_trim_and_compare(transformed, -1, 10, strict=True)

    df_random_with_nans.trim(limits=(-1, 10), inplace=True, relative=False, inclusive=(True, True))
    assert (df_random_with_nans.iloc[1, :-1] == -1).all().all()
    _assert_trim_and_compare(df_random_with_nans, -1, 10)


def test_winsorizer(df_random_with_nans):
    low = df_random_with_nans.quantile(0.01, axis=0)
    high = df_random_with_nans.quantile(0.99, axis=0)

    transformed = df_random_with_nans.winsorize(limits=(0.01, 0.01), inplace=False, inclusive=(True, True))
    assert transformed.shape == df_random_with_nans.shape
    _assert_trim_and_compare(transformed, low, high)

    df_random_with_nans.iloc[0, :-1] = 0
    df_random_with_nans.iloc[1, :-1] = -1

    transformed = df_random_with_nans.winsorize(limits=(-1, 10), inplace=False, relative=False,
                                                inclusive=(False, False))
    _assert_trim_and_compare(transformed, -1, 10, strict=True)

    df_random_with_nans.winsorize(limits=(-1, 10), inplace=True, relative=False, inclusive=(True, True))
    assert (df_random_with_nans.iloc[1, :-1] == -1).all().all()
    _assert_trim_and_compare(df_random_with_nans, -1, 10)


def _assert_trim_and_compare(df, low, high, strict=False):
    assert (df['categorical'] == 'a').all().all()
    df.drop('categorical', axis=1, inplace=True)
    df.impute(inplace=True, strategy='median')
    assert (df > low).all().all() if strict else (df >= low).all().all()
    assert (df < high).all().all() if strict else (df <= high).all().all()


def test_standard_scaler(df_random_unbalanced):
    transformed = df_random_unbalanced.scale(inplace=False, fast_transform=True)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], StandardScaler)
    assert ((transformed.std() - 1).abs() < 1e-2).all()
    assert ((transformed.mean()).abs() < 1e-9).all()
    assert not transformed.equals(df_random_unbalanced)

    df_parallel = df_random_unbalanced.copy(deep=True)
    df_random_unbalanced.scale(inplace=True)
    assert df_random_unbalanced.pipeline._transformations
    assert isinstance(df_random_unbalanced.pipeline._transformations[0], StandardScaler)
    assert ((df_random_unbalanced.std() - 1).abs() < 1e-2).all()
    assert ((df_random_unbalanced.mean()).abs() < 1e-9).all()

    df_parallel['target'] = df_parallel['col_1'] + df_parallel['col_2']
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    transformed = df_parallel.targetize('target').scale(inplace=False, auto=True, exclude={'static'})

    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], StandardScaler)
    assert (transformed['target'] == df_parallel['target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    for col in df_random_unbalanced.columns:
        assert ((transformed[col] - df_random_unbalanced[col]).abs() < 1e-9).all()


def test_maxabs_scaler(df_random_unbalanced):
    df_random_unbalanced['zeros'] = np.zeros(df_random_unbalanced.shape[0])
    transformed = df_random_unbalanced.scale(kind='maxabs', inplace=False, fast_transform=True)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], MaxAbsScaler)

    df_parallel = df_random_unbalanced.copy(deep=True)
    df_random_unbalanced.scale(kind='maxabs', inplace=True)
    assert df_random_unbalanced.pipeline._transformations
    assert isinstance(df_random_unbalanced.pipeline._transformations[0], MaxAbsScaler)
    assert approximately_equal(transformed, df_random_unbalanced)
    assert ((df_random_unbalanced.abs().max() - 1).iloc[:-1].abs() < 1e-5).all()

    df_parallel['target'] = df_parallel['col_1'] + df_parallel['col_2']
    df_parallel['second_target'] = df_parallel['col_3'] + df_parallel['col_4']
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    transformed = df_parallel.targetize(('target', 'second_target')).scale(kind='maxabs', inplace=False, auto=True,
                                                                           exclude={'static'})

    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], MaxAbsScaler)
    assert (transformed['target'] == df_parallel['target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['second_target'] == df_parallel['second_target']).all()
    assert (transformed['static'] == 2).all()
    for col in df_random_unbalanced.columns:
        assert ((transformed[col] - df_random_unbalanced[col]).abs() < 1e-9).all()


def test_robust_scaler(df_random_unbalanced):
    df_random_unbalanced['ones'] = np.ones(df_random_unbalanced.shape[0])
    transformed = df_random_unbalanced.scale(kind='robust', inplace=False, fast_transform=True)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], RobustScaler)

    df_parallel = df_random_unbalanced.copy(deep=True)
    df_random_unbalanced.scale(kind='robust', inplace=True)
    assert df_random_unbalanced.pipeline._transformations
    assert isinstance(df_random_unbalanced.pipeline._transformations[0], RobustScaler)
    assert approximately_equal(transformed, df_random_unbalanced)
    assert (((df_random_unbalanced.quantile(0.75) - df_random_unbalanced.quantile(0.25)) - 1).iloc[
            :-1].abs() < 1e-5).all()

    df_parallel['target'] = df_parallel['col_1'] + df_parallel['col_2']
    df_parallel['second_target'] = df_parallel['col_3'] + df_parallel['col_4']
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    transformed = df_parallel.targetize(('target', 'second_target')).scale(kind='robust', inplace=False, auto=True,
                                                                           exclude={'static'})

    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], RobustScaler)
    assert (transformed['target'] == df_parallel['target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['second_target'] == df_parallel['second_target']).all()
    assert (transformed['static'] == 2).all()
    for col in df_random_unbalanced.columns:
        assert ((transformed[col] - df_random_unbalanced[col]).abs() < 1e-9).all()


def test_minmax_scaler(df_random_unbalanced):
    df_random_unbalanced['zeros'] = np.zeros(df_random_unbalanced.shape[0])
    transformed = df_random_unbalanced.scale(kind='minmax', fast_transform=True, inplace=False)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], MinMaxScaler)

    df_parallel = df_random_unbalanced.copy(deep=True)
    df_random_unbalanced.scale(kind='minmax', inplace=True)

    assert approximately_equal(df_random_unbalanced, transformed)
    assert df_random_unbalanced.pipeline._transformations
    assert isinstance(df_random_unbalanced.pipeline._transformations[0], MinMaxScaler)
    assert (df_random_unbalanced <= 1).all().all()
    assert (df_random_unbalanced >= 0).all().all()
    assert ((1 - df_random_unbalanced.max()).iloc[:-1] < 1e-5).all()

    df_parallel['target'] = df_parallel['col_1'] + df_parallel['col_2']
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    include_cols = ['col_1', 'col_2']
    transformed = df_parallel.targetize('target').scale(kind='minmax', inplace=False, auto=True,
                                                        include=include_cols)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], MinMaxScaler)
    assert (transformed['target'] == df_parallel['target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    for col in include_cols:
        assert ((transformed[col] - df_random_unbalanced[col]).abs() < 1e-9).all()


def test_normalizer(df_random_categorical):
    num_columns = list(df_random_categorical.select_dtypes(np.number).columns)
    trgt = df_random_categorical['num_col_0'] * df_random_categorical['num_col_1']
    df_random_categorical['target'] = trgt
    df_random_categorical.targetize('target')

    transformed = df_random_categorical.normalize(auto=True)
    assert (transformed.columns == df_random_categorical.columns).all()
    assert (transformed['target'] == trgt).all()
    selection = transformed.select_dtypes(np.number)
    selection.drop('target', axis=1, inplace=True)
    assert list(selection.columns) == num_columns
    assert approximately_equal(selection.std(ddof=0, axis=1), 1.0)

    transformed = df_random_categorical.normalize(norm='l1', auto=True)
    selection = transformed.select_dtypes(np.number).drop('target', axis=1)
    assert approximately_equal(selection.abs().sum(axis=1), 1.0)

    df_random_categorical.normalize(norm='max', auto=False, inplace=True, include=num_columns)
    assert df_random_categorical['target'].equals(trgt)
    selection = df_random_categorical.select_dtypes(np.number).drop('target', axis=1)
    assert approximately_equal(selection.abs().max(axis=1), 1)


@pytest.mark.parametrize('strategy', ('mean', 'median', 'most_frequent', 'constant'))
def test_simple_imputer(df_missing, strategy):
    df_orig = df_missing.copy(deep=True)
    dtypes = list(df_missing.dtypes)
    fill_value = -100 if strategy == 'constant' else None

    sklearn_transformed = SklearnSimpleImputer(strategy=strategy, fill_value=fill_value).fit_transform(df_missing)

    transformed = df_missing.impute(strategy=strategy, fill_value=fill_value, inplace=False)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], SimpleImputer)
    assert (transformed.isna().sum() == 0).all()
    assert approximately_equal(transformed, sklearn_transformed)
    assert dtypes == list(transformed.dtypes)

    df_parallel = df_missing.copy(deep=True)
    df_missing.impute(strategy=strategy, fill_value=fill_value, inplace=True)
    assert df_missing.pipeline._transformations
    assert isinstance(df_missing.pipeline._transformations[0], SimpleImputer)
    assert (df_missing.isna().sum() == 0).all()
    assert dtypes == list(df_missing.dtypes)

    df_parallel['_target'] = df_parallel['c'] * 7
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    include_cols = ['a', 'b']
    df_parallel.targetize('_target')
    dtypes = list(df_parallel.dtypes)
    transformed = df_parallel.impute(strategy=strategy, fill_value=fill_value, kind='simple', inplace=False,
                                     auto=True,
                                     include=include_cols)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], SimpleImputer)
    assert (transformed['_target'] == df_parallel['_target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    assert dtypes == list(transformed.dtypes)
    for col in include_cols:
        assert ((transformed[col] - df_missing[col]).abs() < 1e-9).all()

    df_orig.iloc[1, 0] = 7
    df_new = df_orig.stream(df_missing)
    assert df_new.iloc[1, 0] == 7
    assert list(df_new.dtypes) == list(df_orig.dtypes)
    assert approximately_equal(df_missing.iloc[:, 1:], df_new.iloc[:, 1:])
    assert (df_new.isna().sum() == 0).all()
    assert df_new.pipeline._transformations
    assert isinstance(df_new.pipeline._transformations[0], SimpleImputer)


def test_simple_imputer_with_default_strategy(df_missing_obj):
    df_missing = df_missing_obj
    strategy = 'default'
    df_orig = df_missing.copy(deep=True)
    dtypes = list(df_missing.dtypes)

    cat_cols = [col for col in df_missing.columns if df_missing[col].dtype.kind in "OUSVb"]
    other_cols = list(set(df_missing.columns).difference(cat_cols))
    cat_cols = [df_missing.columns.get_loc(col) for col in cat_cols]
    other_cols = [df_missing.columns.get_loc(col) for col in other_cols]

    sklearn_transformed_median = SklearnSimpleImputer(strategy='median').fit_transform(df_missing.iloc[:, other_cols])
    sklearn_transformed_mode = SklearnSimpleImputer(strategy='most_frequent').fit_transform(df_missing.iloc[:, cat_cols])

    transformed = df_missing.impute(kind='simple', strategy=strategy, inplace=False)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], SimpleImputer)
    assert (transformed.isna().sum() == 0).all()
    assert all(transformed.iloc[:, cat_cols] == sklearn_transformed_mode)
    assert approximately_equal(transformed.iloc[:, other_cols], sklearn_transformed_median)
    assert dtypes == list(transformed.dtypes)

    df_parallel = df_missing.copy(deep=True)
    df_missing.impute(strategy=strategy, inplace=True)
    assert df_missing.pipeline._transformations
    assert isinstance(df_missing.pipeline._transformations[0], SimpleImputer)
    assert (df_missing.isna().sum() == 0).all()
    assert dtypes == list(df_missing.dtypes)

    df_parallel['_target'] = df_parallel['c'] * 7
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    include_cols = ['a', 'b']
    df_parallel.targetize('_target')
    dtypes = list(df_parallel.dtypes)
    transformed = df_parallel.impute(strategy=strategy, kind='simple', inplace=False,
                                     auto=True,
                                     include=include_cols)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], SimpleImputer)
    assert (transformed['_target'] == df_parallel['_target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    assert dtypes == list(transformed.dtypes)
    for col in include_cols:
        assert ((transformed[col] - df_missing[col]).abs() < 1e-9).all()

    df_orig.iloc[1, 0] = 7
    df_new = df_orig.stream(df_missing)
    assert df_new.iloc[1, 0] == 7
    assert list(df_new.dtypes) == list(df_orig.dtypes)
    assert all(df_missing.iloc[:, 1:] == df_new.iloc[:, 1:])
    assert (df_new.isna().sum() == 0).all()
    assert df_new.pipeline._transformations
    assert isinstance(df_new.pipeline._transformations[0], SimpleImputer)


@pytest.mark.parametrize('strategy', ('mean', 'median', 'most_frequent', 'constant'))
def test_simple_imputer_object(df_missing_obj, strategy):
    df_orig = df_missing_obj.copy(deep=True)
    df_parallel = df_missing_obj.copy(deep=True)
    df_parallel['_target'] = np.ones(df_parallel.shape[0]) * 8
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    include_cols = ['a', 'b']
    df_parallel.targetize('_target')
    dtypes = list(df_parallel.dtypes)
    transformed = df_parallel.impute(strategy=strategy, kind='simple', inplace=False,
                                     include=include_cols)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], SimpleImputer)
    assert (transformed['_target'] == df_parallel['_target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    assert (transformed.loc[:, include_cols].isna().sum() == 0).all()
    assert dtypes == list(transformed.dtypes)

    df_orig.iloc[1, 0] = 7
    df_new = df_orig.stream(transformed)
    assert df_new.iloc[1, 0] == 7
    assert list(df_new.dtypes) == list(df_orig.dtypes)
    assert (df_new.loc[:, include_cols].isna().sum() == 0).all()
    assert approximately_equal(transformed.loc[2:, include_cols], df_new.loc[2:, include_cols])
    assert df_new.pipeline._transformations
    assert isinstance(df_new.pipeline._transformations[1], SimpleImputer)

    if strategy in ['constant']:
        include_cols = ['c', 'd']
        transformed = transformed.impute(strategy=strategy, kind='simple', inplace=False,
                                         include=include_cols)
        assert transformed.pipeline._transformations
        assert isinstance(transformed.pipeline._transformations[2], SimpleImputer)
        assert (transformed['_target'] == df_parallel['_target']).all()
        assert (transformed['objects'] == df_parallel['objects']).all()
        assert (transformed['static'] == 2).all()
        assert dtypes == list(transformed.dtypes)
        assert (transformed.loc[:, include_cols].isna().sum() == 0).all()


def test_simple_imputer_fill_value(df_missing, df_missing_obj):
    df_orig = df_missing_obj.copy(deep=True)
    strategy = 'constant'
    for i, df_missing in enumerate([df_missing, df_missing_obj]):
        fill_value_list = [-1, -2, "a", "c", -5, -6] if i == 1 else [-1, -2, -3, -4, -5, -6]
        fill_value = {col: value for col, value in zip(df_missing.columns, fill_value_list)}
        df_missing = df_missing.copy(deep=True)
        dtypes = list(df_missing.dtypes)
        transformed = df_missing.impute(strategy=strategy, fill_value=fill_value, inplace=False)
        assert transformed.pipeline._transformations
        assert isinstance(transformed.pipeline._transformations[0], SimpleImputer)
        assert (transformed.isna().sum() == 0).all()
        assert dtypes == list(transformed.dtypes)

        df_parallel = df_missing.copy(deep=True)
        df_missing.impute(strategy=strategy, fill_value=fill_value, inplace=True)
        assert df_missing.pipeline._transformations
        assert isinstance(df_missing.pipeline._transformations[0], SimpleImputer)
        assert (df_missing.isna().sum() == 0).all()
        assert dtypes == list(df_missing.dtypes)

        df_parallel['_target'] = df_parallel['c'] * 7
        df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
        df_parallel['objects'] = ['a'] * df_parallel.shape[0]
        include_cols = ['a', 'b']
        df_parallel.targetize('_target')
        dtypes = list(df_parallel.dtypes)
        fill_value = {'a': -1, 'b': -2}
        transformed = df_parallel.impute(strategy=strategy, fill_value=fill_value, kind='simple', inplace=False,
                                         auto=True,
                                         include=include_cols)
        assert transformed.pipeline._transformations
        assert isinstance(transformed.pipeline._transformations[1], SimpleImputer)
        assert (transformed['_target'] == df_parallel['_target']).all()
        assert (transformed['objects'] == df_parallel['objects']).all()
        assert (transformed['static'] == 2).all()
        assert dtypes == list(transformed.dtypes)
        for col in include_cols:
            assert ((transformed[col] - df_missing[col]).abs() < 1e-9).all()

        df_orig.iloc[1, 0] = 7
        df_new = df_orig.stream(transformed)
        assert list(df_new.dtypes) == list(df_orig.dtypes)
        assert df_new.iloc[1, 0] == 7
        assert (df_new.loc[:, include_cols].isna().sum() == 0).all()
        assert approximately_equal(transformed.loc[2:, include_cols], df_new.loc[2:, include_cols])
        assert df_new.pipeline._transformations
        assert isinstance(df_new.pipeline._transformations[1], SimpleImputer)


@pytest.mark.parametrize('strategy', ('mean', 'median', 'most_frequent', 'constant'))
def test_simple_imputer_blockwise(df_missing_blockwise, strategy):
    df_orig = df_missing_blockwise.copy(deep=True)
    df_missing = df_missing_blockwise.copy(deep=True)
    dtypes = list(df_missing.dtypes)
    fill_value = -100 if strategy == 'constant' else None

    sklearn_transformed = SklearnSimpleImputer(strategy=strategy, fill_value=fill_value).fit_transform(df_missing)

    transformed = df_missing.impute(strategy=strategy, fill_value=fill_value, inplace=False)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], SimpleImputer)
    assert (transformed.isna().sum() == 0).all()
    assert approximately_equal(transformed, sklearn_transformed)
    assert dtypes == list(transformed.dtypes)

    df_missing = df_missing_blockwise
    transformed = df_missing.impute(strategy=strategy, fill_value=fill_value, inplace=False)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], SimpleImputer)
    assert (transformed.isna().sum() == 0).all()
    assert approximately_equal(transformed, sklearn_transformed)
    assert dtypes == list(transformed.dtypes)

    df_orig.iloc[1, 0] = 7
    df_new = df_orig.stream(transformed)
    assert df_new.iloc[1, 0] == 7
    assert approximately_equal(transformed.iloc[:, 1:], df_new.iloc[:, 1:])
    assert (df_new.isna().sum() == 0).all()
    assert df_new.pipeline._transformations
    assert isinstance(df_new.pipeline._transformations[0], SimpleImputer)


@pytest.mark.parametrize('strategy', ('mean', 'median', 'most_frequent', 'constant'))
def test_simple_imputer_nan(df_missing_nan, strategy):
    df_orig = df_missing_nan.copy(deep=True)
    df_missing = df_missing_nan
    dtypes = list(df_missing.dtypes)
    fill_value = -100 if strategy == 'constant' else None

    sklearn_transformed = SklearnSimpleImputer(strategy=strategy, fill_value=fill_value).fit_transform(df_missing)

    transformed = df_missing.impute(strategy=strategy, fill_value=fill_value, inplace=False)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], SimpleImputer)
    if strategy == 'constant':
        assert (transformed.isna().sum() == 0).all()
        assert approximately_equal(transformed, sklearn_transformed)
    else:
        assert (transformed.isna().sum() == 0).sum() == df_missing.shape[1] - 1
        assert approximately_equal(transformed.drop(columns=["f"]), sklearn_transformed)

    assert dtypes == list(transformed.dtypes)

    df_parallel = df_missing.copy(deep=True)
    df_missing.impute(strategy=strategy, fill_value=fill_value, inplace=True)
    assert df_missing.pipeline._transformations
    assert isinstance(df_missing.pipeline._transformations[0], SimpleImputer)
    if strategy == 'constant':
        assert (df_missing.isna().sum() == 0).all()
    else:
        assert (df_missing.isna().sum() == 0).sum() == df_missing.shape[1] - 1
    assert dtypes == list(df_missing.dtypes)

    df_parallel['_target'] = df_parallel['c'] * 7
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    include_cols = ['a', 'b']
    df_parallel.targetize('_target')
    dtypes = list(df_parallel.dtypes)
    transformed = df_parallel.impute(strategy=strategy, fill_value=fill_value, kind='simple', inplace=False,
                                     auto=True,
                                     include=include_cols)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], SimpleImputer)
    assert (transformed['_target'] == df_parallel['_target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    assert dtypes == list(transformed.dtypes)
    for col in include_cols:
        assert ((transformed[col] - df_missing[col]).abs() < 1e-9).all()

    df_orig.iloc[1, 0] = 7
    df_new = df_orig.stream(transformed)
    assert df_new.iloc[1, 0] == 7
    assert approximately_equal(transformed.loc[2:, include_cols], df_new.loc[2:, include_cols])
    assert (df_new.loc[:, include_cols].isna().sum() == 0).all()
    assert df_new.pipeline._transformations
    assert isinstance(df_new.pipeline._transformations[1], SimpleImputer)


@pytest.mark.parametrize('strategy', ('mean', 'median', 'most_frequent', 'constant'))
def test_iterative_imputer(df_missing, strategy):
    df_orig = df_missing.copy(deep=True)
    dtypes = list(df_missing.dtypes)
    sklearn_transformed = SklearnIterativeImputer(initial_strategy=strategy).fit_transform(df_missing)

    transformed = df_missing.impute(initial_strategy=strategy, kind='iterative', inplace=False)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], IterativeImputer)
    assert (transformed.isna().sum() == 0).all()
    assert dtypes == list(transformed.dtypes)
    assert approximately_equal(transformed, sklearn_transformed, 1e-1)

    df_parallel = df_missing.copy(deep=True)
    df_missing.impute(initial_strategy=strategy, kind='iterative', inplace=True)
    assert df_missing.pipeline._transformations
    assert isinstance(df_missing.pipeline._transformations[0], IterativeImputer)
    assert (df_missing.isna().sum() == 0).all()
    assert dtypes == list(df_missing.dtypes)

    df_parallel['_target'] = df_parallel['c'] * 7
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    include_cols = ['a', 'b']
    df_parallel.targetize('_target')
    dtypes = list(df_parallel.dtypes)
    transformed = df_parallel.impute(initial_strategy=strategy, kind='iterative', inplace=False, auto=True,
                                     include=include_cols)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], IterativeImputer)
    assert (transformed['_target'] == df_parallel['_target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    assert dtypes == list(transformed.dtypes)

    df_orig.iloc[1, 0] = 7
    df_new = df_orig.stream(transformed)
    assert list(df_new.dtypes) == list(df_orig.dtypes)
    assert df_new.iloc[1, 0] == 7
    assert (df_new.loc[:, include_cols].isna().sum() == 0).all()
    assert approximately_equal(transformed.loc[2:, include_cols], df_new.loc[2:, include_cols])
    assert df_new.pipeline._transformations
    assert isinstance(df_new.pipeline._transformations[1], IterativeImputer)


@pytest.mark.parametrize('strategy', ('mean', 'median', 'most_frequent', 'constant'))
def test_iterative_imputer_object(df_missing_obj, strategy):
    df_orig = df_missing_obj.copy(deep=True)
    dtypes = list(df_missing_obj.dtypes)
    numeric_cols = df_missing_obj.select_dtypes(include=np.number).columns
    transformed = df_missing_obj.impute(initial_strategy=strategy, kind='iterative', inplace=False)

    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], IterativeImputer)
    assert (transformed.loc[:, numeric_cols].isna().sum() == 0).all()
    assert dtypes == list(transformed.dtypes)

    df_parallel = df_missing_obj.copy(deep=True)
    df_missing_obj.impute(initial_strategy=strategy, kind='iterative', inplace=True)
    assert df_missing_obj.pipeline._transformations
    assert isinstance(df_missing_obj.pipeline._transformations[0], IterativeImputer)
    assert (df_missing_obj.loc[:, numeric_cols].isna().sum() == 0).all()
    assert dtypes == list(df_missing_obj.dtypes)

    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['_target'] = df_parallel['static'] * 7
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    include_cols = ['a', 'b']
    df_parallel.targetize('_target')
    dtypes = list(df_parallel.dtypes)
    transformed = df_parallel.impute(initial_strategy=strategy, kind='iterative', inplace=False, auto=True,
                                     include=include_cols)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], IterativeImputer)
    assert (transformed['_target'] == df_parallel['_target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    assert dtypes == list(transformed.dtypes)

    df_orig.iloc[1, 0] = 7
    df_new = df_orig.stream(transformed)
    assert list(df_new.dtypes) == list(df_orig.dtypes)
    assert df_new.iloc[1, 0] == 7
    assert (df_new.loc[:, include_cols].isna().sum() == 0).all()
    assert approximately_equal(transformed.loc[2:, include_cols], df_new.loc[2:, include_cols])
    assert df_new.pipeline._transformations
    assert isinstance(df_new.pipeline._transformations[1], IterativeImputer)


@pytest.mark.parametrize('strategy', ('mean', 'median', 'most_frequent', 'constant'))
def test_iterative_imputer_nan(df_missing_nan, strategy):
    df_orig = df_missing_nan.copy(deep=True)
    df_missing = df_missing_nan
    dtypes = list(df_missing.dtypes)
    if strategy != 'constant':
        sklearn_transformed = SklearnIterativeImputer(initial_strategy=strategy).fit_transform(df_missing)

    transformed = df_missing.impute(initial_strategy=strategy, kind='iterative', inplace=False)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], IterativeImputer)
    assert (transformed.isna().sum() == 0).all()
    # assert approximately_equal(transformed.drop(columns=["f"]), sklearn_transformed, 1e-1)
    assert dtypes == list(transformed.dtypes)

    df_parallel = df_missing.copy(deep=True)
    df_missing.impute(initial_strategy=strategy, kind='iterative', inplace=True)
    assert df_missing.pipeline._transformations
    assert isinstance(df_missing.pipeline._transformations[0], IterativeImputer)
    assert (df_missing.isna().sum() == 0).all()
    assert dtypes == list(df_missing.dtypes)

    df_parallel['_target'] = df_parallel['c'] * 7
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    include_cols = ['a', 'b']
    df_parallel.targetize('_target')
    dtypes = list(df_parallel.dtypes)
    transformed = df_parallel.impute(initial_strategy=strategy, kind='iterative', inplace=False, auto=True,
                                     include=include_cols)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], IterativeImputer)
    assert (transformed['_target'] == df_parallel['_target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    assert dtypes == list(transformed.dtypes)

    df_orig.iloc[1, 0] = 7
    df_new = df_orig.stream(transformed)
    assert list(df_new.dtypes) == list(df_orig.dtypes)
    assert df_new.iloc[1, 0] == 7
    assert (df_new.loc[:, include_cols].isna().sum() == 0).all()
    assert approximately_equal(transformed.loc[2:, include_cols], df_new.loc[2:, include_cols])
    assert df_new.pipeline._transformations
    assert isinstance(df_new.pipeline._transformations[1], IterativeImputer)


def test_knn_imputer(df_missing):
    df_orig = df_missing.copy(deep=True)
    dtypes = list(df_missing.dtypes)
    sklearn_transformed = SklearnKNNImputer().fit_transform(df_missing)

    transformed = df_missing.impute(kind='knn', inplace=False)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], KNNImputer)
    assert (transformed.isna().sum() == 0).all()
    assert dtypes == list(transformed.dtypes)
    assert approximately_equal(transformed, sklearn_transformed, 1e-1)

    df_parallel = df_missing.copy(deep=True)
    df_missing.impute(kind='knn', inplace=True)
    assert df_missing.pipeline._transformations
    assert isinstance(df_missing.pipeline._transformations[0], KNNImputer)
    assert (df_missing.isna().sum() == 0).all()
    assert dtypes == list(df_missing.dtypes)

    df_parallel['_target'] = df_parallel['c'] * 7
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    include_cols = ['a', 'b']
    df_parallel.targetize('_target')
    dtypes = list(df_parallel.dtypes)
    transformed = df_parallel.impute(kind='knn', inplace=False, auto=True, include=include_cols)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], KNNImputer)
    assert (transformed['_target'] == df_parallel['_target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    assert dtypes == list(transformed.dtypes)

    df_orig.iloc[1, 0] = 7
    df_new = df_orig.stream(transformed)
    assert list(df_new.dtypes) == list(df_orig.dtypes)
    assert df_new.iloc[1, 0] == 7
    assert (df_new.loc[:, include_cols].isna().sum() == 0).all()
    assert approximately_equal(transformed.loc[2:, include_cols], df_new.loc[2:, include_cols])
    assert df_new.pipeline._transformations
    assert isinstance(df_new.pipeline._transformations[1], KNNImputer)


def test_knn_imputer_object(df_missing_obj):
    df_orig = df_missing_obj.copy(deep=True)
    dtypes = list(df_missing_obj.dtypes)
    numeric_cols = df_missing_obj.select_dtypes(include=np.number).columns
    transformed = df_missing_obj.impute(kind='knn', inplace=False)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], KNNImputer)
    assert (transformed.loc[:, numeric_cols].isna().sum() == 0).all()
    assert dtypes == list(transformed.dtypes)

    df_parallel = df_missing_obj.copy(deep=True)
    df_missing_obj.impute(kind='knn', inplace=True)
    assert df_missing_obj.pipeline._transformations
    assert isinstance(df_missing_obj.pipeline._transformations[0], KNNImputer)
    assert (df_missing_obj.loc[:, numeric_cols].isna().sum() == 0).all()
    assert dtypes == list(df_missing_obj.dtypes)

    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['_target'] = df_parallel['static'] * 7
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    include_cols = ['a', 'b']
    df_parallel.targetize('_target')
    dtypes = list(df_parallel.dtypes)
    transformed = df_parallel.impute(kind='knn', inplace=False, auto=True,
                                     include=include_cols)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], KNNImputer)
    assert (transformed['_target'] == df_parallel['_target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    assert dtypes == list(transformed.dtypes)

    df_orig.iloc[1, 0] = 7
    df_new = df_orig.stream(transformed)
    assert list(df_new.dtypes) == list(df_orig.dtypes)
    assert df_new.iloc[1, 0] == 7
    assert (df_new.loc[:, include_cols].isna().sum() == 0).all()
    assert approximately_equal(transformed.loc[2:, include_cols], df_new.loc[2:, include_cols])
    assert df_new.pipeline._transformations
    assert isinstance(df_new.pipeline._transformations[1], KNNImputer)


def test_knn_imputer_nan(df_missing_nan):
    df_orig = df_missing_nan.copy(deep=True)
    df_missing = df_missing_nan
    dtypes = list(df_missing.dtypes)
    sklearn_transformed = SklearnKNNImputer().fit_transform(df_missing)

    transformed = df_missing.impute(kind='knn', inplace=False)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[0], KNNImputer)
    assert (transformed.isna().sum() == 0).all()
    # assert approximately_equal(transformed.drop(columns=["f"]), sklearn_transformed, 1e-1)
    assert dtypes == list(transformed.dtypes)

    df_parallel = df_missing.copy(deep=True)
    df_missing.impute(kind='knn', inplace=True)
    assert df_missing.pipeline._transformations
    assert isinstance(df_missing.pipeline._transformations[0], KNNImputer)
    assert (df_missing.isna().sum() == 0).all()
    assert dtypes == list(df_missing.dtypes)

    df_parallel['_target'] = df_parallel['c'] * 7
    df_parallel['static'] = np.ones(df_parallel.shape[0]) * 2
    df_parallel['objects'] = ['a'] * df_parallel.shape[0]
    include_cols = ['a', 'b']
    df_parallel.targetize('_target')
    dtypes = list(df_parallel.dtypes)
    transformed = df_parallel.impute(kind='knn', inplace=False, auto=True, include=include_cols)
    assert transformed.pipeline._transformations
    assert isinstance(transformed.pipeline._transformations[1], KNNImputer)
    assert (transformed['_target'] == df_parallel['_target']).all()
    assert (transformed['objects'] == df_parallel['objects']).all()
    assert (transformed['static'] == 2).all()
    assert dtypes == list(transformed.dtypes)

    df_orig.iloc[1, 0] = 7
    df_new = df_orig.stream(transformed)
    assert list(df_new.dtypes) == list(df_orig.dtypes)
    assert df_new.iloc[1, 0] == 7
    assert (df_new.loc[:, include_cols].isna().sum() == 0).all()
    assert approximately_equal(transformed.loc[2:, include_cols], df_new.loc[2:, include_cols])
    assert df_new.pipeline._transformations
    assert isinstance(df_new.pipeline._transformations[1], KNNImputer)


def test_splitter_sequential(df):
    df['a'] = np.arange(df.shape[0])
    for i in range(6):
        if i == 0:
            split = df.split(kind='sequential', sizes=(0.5, 0.5), inplace=True, pipeline=True, return_indices=True)
        elif i == 1:
            split = df.split(kind='sequential', sizes=[0.5], n_splits=2, inplace=True, pipeline=True,
                             return_indices=True)
        elif i == 2:
            split = df.train_test_split(kind='sequential', train_size=0.5, test_size=None, inplace=True,
                                        pipeline=True,
                                        return_indices=True)
        elif i == 3:
            split = df.train_test_split(kind='sequential', train_size=None, test_size=0.5, inplace=True,
                                        pipeline=True,
                                        return_indices=True)
        elif i == 4:
            split = df.train_test_split(kind='sequential', train_size=0.5, test_size=0.5, inplace=True,
                                        pipeline=True,
                                        return_indices=True)
        else:
            split = df.train_test_split(kind='sequential', sizes=(0.5, 0.5), inplace=True, pipeline=True,
                                        return_indices=True)
        assert len(split) == 2
        assert int(df.loc[split[0], ['a']].nunique()) == int(df.loc[split[1], ['a']].nunique()) == df.shape[0] // 2
        assert len(split[0]) == len(split[1]) == df.shape[0] // 2
        assert max(df.loc[split[0], 'a']) < min(df.loc[split[1], 'a'])
        assert (set(split[0]) & set(split[1])) == set()
        assert set(split[0]).union(set(split[1])) == set(df.index)


def test_splitter_shuffled(df):
    df['a'] = np.arange(df.shape[0])
    for i in range(6):
        if i == 0:
            split = df.split(sizes=(0.5, 0.5), kind='shuffled', inplace=True, pipeline=True, return_indices=True)
        elif i == 1:
            split = df.split(sizes=[0.5], n_splits=2, kind='shuffled', inplace=True, pipeline=True,
                             return_indices=True)
        elif i == 2:
            split = df.train_test_split(train_size=0.5, test_size=None, kind='shuffled', inplace=True,
                                        pipeline=True,
                                        return_indices=True)
        elif i == 3:
            split = df.train_test_split(train_size=None, test_size=0.5, kind='shuffled', inplace=True,
                                        pipeline=True,
                                        return_indices=True)
        elif i == 4:
            split = df.train_test_split(train_size=0.5, test_size=0.5, kind='shuffled', inplace=True, pipeline=True,
                                        return_indices=True)
        else:
            split = df.train_test_split(sizes=(0.5, 0.5), kind='shuffled', inplace=True, pipeline=True,
                                        return_indices=True)
        assert len(split) == 2
        assert int(df.loc[split[0], ['a']].nunique()) == int(df.loc[split[1], ['a']].nunique()) == df.shape[0] // 2
        assert len(split[0]) == len(split[1]) == df.shape[0] // 2
        assert max(df.loc[split[0], 'a']) > min(df.loc[split[1], 'a'])
        assert (set(split[0]) & set(split[1])) == set()
        assert set(split[0]).union(set(split[1])) == set(df.index)


def test_splitter_sort_by(df):
    df['a'] = np.random.permutation(np.arange(df.shape[0]))
    for i in range(6):
        if i == 0:
            split = df.split(sort_by='a', kind='sorted', sizes=(0.5, 0.5), inplace=True, pipeline=True,
                             return_indices=True, ordinal_indices=False)
        elif i == 1:
            split = df.split(sort_by='a', kind='sorted', sizes=[0.5], n_splits=2, inplace=True, pipeline=True,
                             return_indices=True, ordinal_indices=False)
        elif i == 2:
            split = df.train_test_split(sort_by='a', kind='sorted', train_size=0.5, test_size=None, inplace=True,
                                        pipeline=True,
                                        return_indices=True, ordinal_indices=False)
        elif i == 3:
            split = df.train_test_split(sort_by='a', kind='sorted', train_size=None, test_size=0.5, inplace=True,
                                        pipeline=True,
                                        return_indices=True, ordinal_indices=False)
        elif i == 4:
            split = df.train_test_split(sort_by='a', kind='sorted', train_size=0.5, test_size=0.5, inplace=True,
                                        pipeline=True,
                                        return_indices=True, ordinal_indices=False)
        else:
            split = df.train_test_split(sort_by='a', kind='sorted', sizes=(0.5, 0.5), inplace=True, pipeline=True,
                                        return_indices=True, ordinal_indices=False)
        assert len(split) == 2
        assert int(df.loc[split[0], ['a']].nunique()) == int(df.loc[split[1], ['a']].nunique()) == df.shape[0] // 2
        assert len(split[0]) == len(split[1]) == df.shape[0] // 2
        assert max(df.loc[split[0], 'a']) < min(df.loc[split[1], 'a'])
        assert (set(split[0]) & set(split[1])) == set()
        assert set(split[0]).union(set(split[1])) == set(df.index)


@pytest.mark.parametrize(('train_size', 'val_size', 'test_size'),
                         [(0.5, 0.3, 0.2), (0.5, 0.3, None), (None, 0.3, 0.2),
                          (0.5, None, 0.2)])
def test_train_val_test_splitter_not_stratified(df, train_size, val_size, test_size):
    for i in range(3):
        if i == 0:
            kwargs = {'kind': 'sorted', 'sort_by': 'a'}
            df['a'] = np.random.permutation(np.arange(df.shape[0]))
        elif i == 1:
            kwargs = {'kind': 'sequential'}
            df['a'] = np.arange(df.shape[0])
        elif i == 2:
            kwargs = {'kind': 'shuffled'}
            df['a'] = np.arange(df.shape[0])
        split = df.train_val_test_split(train_size=train_size, val_size=val_size, test_size=test_size, inplace=True,
                                        pipeline=True, return_indices=True, ordinal_indices=False, **kwargs)
        train_size, val_size, test_size = 0.5, 0.3, 0.2
        assert int(df.loc[split[0], ['a']].nunique()) / train_size == int(
            df.loc[split[1], ['a']].nunique()) / val_size == \
               int(df.loc[split[2], ['a']].nunique()) / test_size == df.shape[0]
        assert len(split[0]) / train_size == len(split[1]) / val_size == len(split[2]) / test_size == df.shape[0]
        if i == 2:
            assert max(df.loc[split[0], 'a']) > min(df.loc[split[1], 'a'])
            assert max(df.loc[split[1], 'a']) > min(df.loc[split[2], 'a'])
        else:
            assert max(df.loc[split[0], 'a']) < min(df.loc[split[1], 'a'])
            assert max(df.loc[split[1], 'a']) < min(df.loc[split[2], 'a'])
        assert (set(split[0]) & set(split[1])) == set()
        assert (set(split[0]) & set(split[2])) == set()
        assert (set(split[1]) & set(split[2])) == set()
        assert set(split[0]).union(set(split[1])).union(set(split[2])) == set(df.index)


def test_splitters_with_sizes_none(df):
    df['a'] = np.arange(df.shape[0])
    split = df.train_test_split(train_size=None, test_size=None, return_indices=True, ordinal_indices=False)
    assert int(df.loc[split[0], ['a']].nunique()) / 0.8 == int(df.loc[split[1], ['a']].nunique()) / 0.2 == df.shape[
        0]
    split = df.train_val_test_split(train_size=None, test_size=None, val_size=None, ordinal_indices=False,
                                    return_indices=True)
    assert int(df.loc[split[0], ['a']].nunique()) / 0.8 == int(df.loc[split[1], ['a']].nunique()) / 0.1 == \
           int(df.loc[split[2], ['a']].nunique()) / 0.1 == df.shape[0]


def test_splitters_returning_dfs(df):
    def assertions(df, split, sizes):
        for i in range(len(split)):
            assert isinstance(split[i], pd.DataFrame)
            assert split[i].shape[0] == df.shape[0] * sizes[i]

    split = df.train_test_split(ordinal_indices=False, train_size=None, test_size=None)
    assertions(df, split, (0.8, 0.2))
    split = df.train_val_test_split(ordinal_indices=False, train_size=None, test_size=None, val_size=None)
    assertions(df, split, (0.8, 0.1, 0.1))


def assertions_stratify(df, split, sizes, target, p, eps):
    for i in range(len(sizes)):
        assert len(split[i]) / sizes[i] == df.shape[0]
        for j in range(len(sizes)):
            if i != j:
                assert (set(split[i]) & set(split[j])) == set()
        for val in target:
            assert approximately_equal_scalars((df.loc[split[i], 'a'] == val).sum(), int(len(split[i]) * p[val]),
                                               eps)
    assert set().union(*split) == set(df.index)


def test_splitters_stratify(df_random, df):
    target = [0, 1, 2, 3]
    p = [0.2, 0.3, 0.3, 0.2]
    eps = 2

    for df in [df_random, df]:
        target_col = np.repeat(target, [int(p[i] * df.shape[0]) for i in range(len(p))])
        df['a'] = np.random.permutation(target_col)
        for i in range(2):
            train_size = 0.5
            test_size = 0.5
            if i == 0:
                split = df.train_test_split(kind='stratified', stratify_by='a', train_size=train_size,
                                            test_size=test_size,
                                            ordinal_indices=False, return_indices=True)
            elif i == 1:
                df.targetize('a')
                split = df.train_test_split(kind='stratified', train_size=train_size, test_size=test_size,
                                            ordinal_indices=False, return_indices=True)
            assertions_stratify(df, split, [train_size, test_size], target, p, eps)
        train_size = 0.5
        val_size = 0.3
        test_size = 0.2
        split = df.train_val_test_split(kind='stratified', train_size=train_size, val_size=val_size,
                                        test_size=test_size,
                                        ordinal_indices=False, return_indices=True)
        assertions_stratify(df, split, [train_size, val_size, test_size], target, p, eps)
        sizes = (0.5, 0.3, 0.1, 0.1)
        split = df.split(kind='stratified', sizes=sizes, ordinal_indices=False, return_indices=True)
        assertions_stratify(df, split, sizes, target, p, eps)


def assertions(df, split, sizes):
    for n in range(len(split)):
        assert split[n].shape[0] == df.shape[0] * sizes[n]
        for m in range(len(sizes)):
            if n != m:
                assert (set(split[n]) & set(split[m])) == set()
    assert set().union(*split) == set(df.index)


def assertions_df(df, split, sizes):
    for n in range(len(split)):
        assert split[n].shape[0] == df.shape[0] * sizes[n]
        for m in range(len(sizes)):
            if n != m:
                assert (set(split[n].index) & set(split[m].index)) == set()
    assert set().union(*[spl.index for spl in split]) == set(df.index)


def test_splitters_array_split(df):
    df['a'] = np.arange(df.shape[0])
    for kind in ['sequential', 'shuffled']:
        for n_splits in [5, 10, 20]:
            split = df.split(kind=kind, n_splits=n_splits, ordinal_indices=False, return_indices=True)
            assertions(df, split, [1 / n_splits] * n_splits)
            if kind == 'shuffled':
                for i in range(len(split) - 1):
                    assert max(df.loc[split[i], 'a']) > min(df.loc[split[i + 1], 'a'])
            else:
                for i in range(len(split) - 1):
                    assert max(df.loc[split[i], 'a']) < min(df.loc[split[i + 1], 'a'])


def test_splitters_separation(df):
    df['a'] = np.arange(df.shape[0])
    df['b'] = np.arange(df.shape[0])
    df.targetize('b')
    train_size = 0.5
    test_size = 0.5
    split = df.train_test_split(train_size=train_size, test_size=test_size, ordinal_indices=False, separate=True)
    assertions(df, [spl[0].index for spl in split], [train_size, test_size])
    train_size = 0.5
    test_size = 0.3
    val_size = 0.2
    split = df.train_val_test_split(train_size=train_size, val_size=val_size, test_size=test_size,
                                    ordinal_indices=False, separate=True)
    assertions(df, [spl[0].index for spl in split], [train_size, val_size, test_size])


def assertions_kfold(split, df, k):
    for i in range(len(split)):
        assert len(split[i]) == 2
        assert approximately_equal_scalars(len(split[i][0]), df.shape[0] - df.shape[0] // k, 1)
        assert approximately_equal_scalars(len(split[i][1]), df.shape[0] // k, 1)
        assert (set(split[i][0]) & set(split[i][1])) == set()
        assert set().union(*split[i]) == set(df.index)
        if i < len(split) - 1:
            assert len(set(split[i][0]) & set(split[i + 1][0])) == df.shape[0] // k * (k - 1) or \
                   len(set(split[i][0]) & set(split[i + 1][0])) == df.shape[0] // k * (k - 2)
            assert (set(split[i][1]) & set(split[i + 1][1])) == set()


def test_kfold_split(df, df_random):
    k = 5
    #  shuffled, as_list
    split = df.kfold_split(n_splits=k, return_indices=True, as_list=True)
    assertions_kfold(split, df, k)
    #  shuffled, as generator
    for split in df.kfold_split(n_splits=k, return_indices=True):
        split = [split]
        assertions_kfold(split, df, k)

    #  stratified
    target = [0, 1, 2, 3]
    p = [0.2, 0.3, 0.3, 0.2]
    eps = 2
    target_col = np.repeat(target, [int(p[i] * df.shape[0]) for i in range(len(p))])
    df['a'] = np.random.permutation(target_col)
    df.targetize('a')
    #  as_list
    split = df.kfold_split(kind='stratified', n_splits=k, return_indices=True, as_list=True)
    assertions_kfold(split, df, k)
    for i in range(len(split)):
        assertions_stratify(df, split[i], [len(split[i][0])/df.shape[0], len(split[i][1])/df.shape[0]], target, p, eps)

    #  separately as train and test
    for train, test in df.kfold_split(n_splits=k, return_indices=True):
        assertions_kfold([[train, test]], df, k)

    for train, test in df_random.kfold_split(n_splits=k, return_indices=True):
        assertions_kfold([[train, test]], df_random, k)
