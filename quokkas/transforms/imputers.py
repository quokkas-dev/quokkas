"""
This code draws heavily in both functionality and implementation logic on
scikit-learn:
https://scikit-learn.org/stable/
In particular, the code below was partially taken from scikit-learn and
later modified.

The use of this code is allowed under BSD license. scikit-learn license can
be found below:

BSD 3-Clause License

Copyright (c) 2007-2021 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Please also refer to the "LICENSES" directory of this repository.
"""

from __future__ import annotations

import numbers
import math

import numpy as np
import numpy.ma as ma
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats
from collections import namedtuple
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances_chunked

from ..eda.categorical import CategoricalDetector
from .generic import _BaseProcessor
from ..utils.sk_utils import _get_ordered_idx, _most_frequent, _get_dense_mask, _get_weights, is_scalar_nan


class Imputer(_BaseProcessor):
    """
    Base class for the quokkas implementations of imputers. Supports the following arguments:

    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be imputed too
    :param auto: if True, the transform will only be attempted for numeric columns
    :param inplace: if True, the transform will be completed inplace
    :param missing_values: indicates the missing value in the dataframe that will be replaced
    """

    def __init__(self,
                 include: set | list | None = None,
                 exclude: set | list | None = None,
                 include_target: bool = False,
                 auto: bool = True,
                 inplace: bool = False,
                 missing_values: int | float | str | np.nan | None = np.nan):
        _BaseProcessor.__init__(self, inplace, auto, include, exclude, include_target)
        self.params = None
        self.missing_values = missing_values

    def reset_columns(self, df, Xt, dtypes):
        if len(set(dtypes)) == 1:
            df.loc[:, self.params] = Xt
        else:
            for i, col in enumerate(self.params):
                df.loc[:, col] = Xt[:, i]
                if df.dtypes[col] != dtypes[col]:
                    df.loc[:, col] = df.loc[:, col].astype(dtypes[col])

    def equals(self, other):
        return _BaseProcessor.equals(self, other) \
               and (self.missing_values == other.missing_values) or (is_scalar_nan(self.missing_values) and is_scalar_nan(other.missing_values))


class SimpleImputer(Imputer):
    ALLOWED_STRATEGIES = {'default', 'mean', 'median', 'most_frequent', 'constant'}

    """
    Imputation for completing missing values on a per-column basis using one of the 5 strategies:

    - 'default': numerical columns are imputed using median, while categorical columns are imputed using mode,
    - 'mean': impute missing values using the mean of the column,
    - 'median': impute missing values using the median of the column,
    - 'most_frequent': impute missing values using the most frequent value (mode) in the column,
    - 'constant': impute missing values using a constant value.

    :param strategy: the imputation strategy to use.
    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be imputed too
    :param auto: ignored for SimpleImputer, is provided for consistency
    :param inplace: if True, the transform will be completed inplace
    :param fast_transform: attempts to transform the data as-is -
        in particular, no auto-detection of columns to be transformed will be
        attempted. 'include' or 'exclude' arguments cannot be provided together
        with fast_transform. If a target is provided and 'include_target' is set to
        False, the data won't be transformed 'as-is'. Default False.
    :param missing_values: indicates the missing value in the dataframe that will be replaced.
    :param fill_value: value(s) to use for imputation when strategy='constant'. If provided as dictionary, keys will be
    used to infer columns to be imputed, and include-exlude logic will be ignored.
    :param cat_detection_strategy: 'count', 'type' or 'count&type'. If
    'count', the categorical features will be detected based on the
    number of distinct values (with border being min(cat_share * num_rows,
    cat_number)), if 'type' - based on the type of the column, if
    'count&type' - based on both
    :param cat_number: if 'count' or 'count&type', the features with less
    than this number of distinct values will be considered categorical (if
    cat_share * num_rows is larger than cat_number)
    :param cat_share: if 'count' or 'count&type', the features with less
    than cat_share * num_rows of distinct values will be considered categorical
    (if cat_number is larger than cat_share * num_rows)
    :return: transformed dataframe
    """

    def __init__(self,
                 strategy: str = "default",
                 include: set | list | None = None,
                 exclude: set | list | None = None,
                 include_target: bool = False,
                 auto: bool = False,
                 inplace: bool = False,
                 fast_transform: bool = False,
                 missing_values: int | float | str | np.nan | None = np.nan,
                 fill_value=None,
                 cat_detection_strategy='count',
                 cat_number=CategoricalDetector.DEFAULT_CAT_NUMBER,
                 cat_share=CategoricalDetector.DEFAULT_CAT_SHARE
                 ):
        Imputer.__init__(self, inplace=inplace, include=include, exclude=exclude,
                         include_target=include_target, auto=False, missing_values=missing_values)
        self.blockwise = self.include is None and self.exclude is None
        CategoricalDetector.validate_strategy(cat_detection_strategy)
        self.cat_detection_strategy = CategoricalDetector.ALLOWED_STRATEGIES.index(cat_detection_strategy)
        self.cat_number = cat_number
        self.cat_share = cat_share

        if fill_value is not None and not np.isscalar(fill_value) and not isinstance(fill_value, dict):
            raise ValueError("fill_value must be a scalar or a dictionary.")

        if fast_transform:
            if not self.blockwise:
                raise ValueError('cannot provide include or exclude when using fast transform')

        self.fast_transform = fast_transform

        if strategy not in self.ALLOWED_STRATEGIES:
            raise ValueError(
                "received unknown strategy parameter"
                f"{strategy}. Expected one of:"
                f" {self.ALLOWED_STRATEGIES}"
            )

        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    # check if the second condition is relevant
    def blockwisable(self, df):
        """
        Checks if fit and transform can be performed using blockwise approach, when all columns are selected and
        transformed together as a numpy array (as opposed to the columnwise approach when fit and transform happen for
        each column separately in a for loop).

        :param df: dataframe to be transformed
        :return: True if blockwise fit and transform can be performed, False otherwise
        """
        if self.include is not None or self.exclude is not None or (df.target and self.include_target):
            return False
        dtypes = df.dtypes.unique()
        if dtypes.shape[0] > 1:
            return False
        if dtypes[0].kind in "OUSMmb":
            return False
        if self.strategy == "default":
            return False
        if df.isna().all().any():
            return False
        return True

    def _fit_blockwise(self, df):
        """
        Fits the imputer using blockwise approach.

        :param df: dataframe to be transformed
        """
        self.params = df.columns
        nd = df.to_numpy()

        # TODO: potentially rework logic if missing_values is nan. Don't need masked array in that case
        self.statistics_ = self._get_nd_statistics(nd, multidim=True, strategy=self.strategy)

    def _transform_blockwise(self, df):
        """
        Performs blockwise transform.

        :param df: dataframe to be transformed
        :return: transformed dataframe
        """
        nd = df.to_numpy()
        statistics = self.statistics_
        # compute mask before eliminating invalid features

        missing_mask = _get_dense_mask(nd, self.missing_values)

        # Delete the invalid columns if strategy is not constant
        if self.strategy == "constant":
            valid_statistics = statistics
            valid_statistics_indexes = None
        else:
            # same as np.isnan but also works for object dtypes
            valid_mask = ~_get_dense_mask(statistics, np.nan)
            valid_statistics = statistics[valid_mask]
            valid_statistics_indexes = np.flatnonzero(valid_mask)
            nd = nd[:, valid_statistics_indexes]

        # Do actual imputation
        # use mask computed before eliminating invalid mask
        if valid_statistics_indexes is None:
            mask_valid_features = missing_mask
        else:
            mask_valid_features = missing_mask[:, valid_statistics_indexes]
        n_missing = np.sum(mask_valid_features, axis=0)
        values = np.repeat(valid_statistics, n_missing)
        coordinates = np.where(mask_valid_features.transpose())[::-1]

        nd[coordinates] = values
        return df.reinitialize(nd)

    def _fit_columnwise(self, df):
        """
        Fits the imputer using columnwise approach. If strategy is not 'constant', columns that only contain nans are
        ignored. If strategy is 'mean' or 'median', columns of types object, Unicode, (byte-)string, datetime,
        timedelta, and bool are ignored.

        :param df: dataframe to be transformed
        """
        cols = [col for col in df if col in self._select_columns(df)]
        self.params = []
        statistics = []
        if self.strategy == "default":
            cat_cols = CategoricalDetector.determine_categorical(df, cols, self.cat_detection_strategy,
                                                                 cat_number=self.cat_number,
                                                                 cat_share=self.cat_share)
        if self.strategy == "constant":
            self.params = cols if not isinstance(self.fill_value, dict) else list(self.fill_value.keys())
            if self.fill_value is None:
                statistics = ["missing value" if df[col].dtype.kind in "OUSMmb" else 0 for col in cols]
            elif np.isscalar(self.fill_value):
                statistics = [self.fill_value for _ in cols]
            else:
                if len(self.fill_value) != len(cols):
                    raise ValueError(
                        f"Expected a scalar or a dictionary of length {len(cols)}, got dictionary "
                        f"of length {len(self.fill_value)} instead.")
                statistics = [self.fill_value[col] for col in cols]
        else:
            for i, col in enumerate(cols):
                nd = df[col].to_numpy()
                if df[col].isnull().all() or (nd.dtype.kind in "OUSMmb" and self.strategy in ["mean", "median"]):
                    continue
                self.params.append(col)
                if self.strategy == "default":
                    if nd.dtype.kind in "OUSVb" or col in cat_cols:
                        statistics.append(self._get_nd_statistics(nd, multidim=False, strategy='most_frequent'))
                    else:
                        statistics.append(self._get_nd_statistics(nd, multidim=False, strategy='median'))
                else:
                    statistics.append(self._get_nd_statistics(nd, multidim=False, strategy=self.strategy))

        self.statistics_ = statistics

    def _transform_columnwise(self, df):
        """
        Performs columnwise transform: all columns are transformed separately using a for loop.

        :param df: dataframe to be transformed
        :return: transformed dataframe
        """
        value_to_mask = self.missing_values
        for col, stat in zip(self.params, self.statistics_):
            if is_numeric_dtype(df[col]):
                nd = df[col].to_numpy()
                nd[_get_dense_mask(nd, value_to_mask)] = stat
                df[col] = nd
            else:
                if value_to_mask is pd.NA or (isinstance(value_to_mask, numbers.Real) and math.isnan(value_to_mask)):
                    df[col][df[col].isna()] = stat
                else:
                    df[col][df[col] == value_to_mask] = stat
        return df

    def _transform_columnwise(self, df):
        """
        Performs columnwise transform: all columns are transformed separately using a for loop.

        :param df: dataframe to be transformed
        :return: transformed dataframe
        """
        value_to_mask = self.missing_values
        for col, stat in zip(self.params, self.statistics_):
            nd = df[col].to_numpy()
            nd[_get_dense_mask(nd, value_to_mask)] = stat
            df[col] = nd
        return df

    def fit(self, df):
        """
        Fits the imputer using blockwise or columnwise approach.
        
        :param df: dataframe to be transformed
        """
        self.blockwise = self.blockwisable(df)
        with df.pipeline:
            if self.blockwise:
                self._fit_blockwise(df)
            else:
                self._fit_columnwise(df)
        self.fitted = True

    def transform(self, df):
        """
        Transforms the dataframe using blockwise or columnwise approach.

        :param df: dataframe to be transformed
        :return: transformed dataframe
        """
        self.validate_fit()
        if not self.inplace:
            maybe_df = df.copy(deep=True)
        else:
            maybe_df = df
        with maybe_df.pipeline:
            if self.blockwise and self.fast_transform:
                self._transform_blockwise(maybe_df)
            else:
                self._transform_columnwise(maybe_df)
        maybe_df.pipeline.add(self)
        if not self.inplace:
            return maybe_df

    def _get_nd_statistics(self, nd: np.ndarray, multidim: bool, strategy: str = 'median'):
        """
        Computes statistics (values to be imputed) for the given ndarray, which can be 1d or 2d.

        :param nd: ndarray to compute statistics for
        :param multidim: boolean indicating if nd is 1d (False) or 2d (True)
        :param strategy: strategy to use for computing statistics
        :return: statistics for the given ndarray
        """
        masked_nd = ma.masked_array(nd, mask=_get_dense_mask(nd, self.missing_values))

        # Mean
        if strategy == "mean":
            mean_masked = np.ma.mean(masked_nd, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            mean = np.ma.getdata(mean_masked)
            mean[np.ma.getmask(mean_masked)] = np.nan
            statistics_ = mean

        # Median
        elif strategy == "median":
            median_masked = np.ma.median(masked_nd, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            median = np.ma.getdata(median_masked)
            median[np.ma.getmaskarray(median_masked)] = np.nan

            statistics_ = median

        # Most frequent
        elif strategy == "most_frequent":

            nd = nd.transpose()
            if nd.dtype.kind == "O":
                most_frequent = np.empty(nd.shape[0], dtype=object)
            else:
                most_frequent = np.empty(nd.shape[0])
            if multidim:
                for i, row in enumerate(nd[:]):
                    most_frequent[i] = _most_frequent(row)
            else:
                most_frequent = _most_frequent(nd)

            statistics_ = most_frequent

        else:
            fill_value = 0 if self.fill_value is None else self.fill_value
            if np.isscalar(fill_value):
                statistics_ = np.full(nd.shape[1], fill_value, dtype=nd.dtype) if multidim else fill_value
            else:
                statistics_ = fill_value
        return statistics_ if multidim or not isinstance(statistics_, np.ndarray) else statistics_.item()

    def equals(self, other):
        return Imputer.equals(self, other) \
               and self.fill_value == other.fill_value \
               and self.strategy == other.strategy \
               and self.blockwise == other.blockwise \
               and self.fast_transform == other.fast_transform


class IterativeImputer(Imputer):
    """
    Imputes missing values in a dataframe based on values in other columns. First, an initial imputation is performed
    using SimpleImputer. Afterwards imputation is performed iteratively in multiple rounds. In each round, columns are
    imputed separately using the values in other columns, whereas the order of columns is determined by the parameter
    'imputation_order'.

    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be imputed too
    :param auto: ignored for SimpleImputer, is provided for consistency
    :param inplace: if True, the transform will be completed inplace
    :param estimator: The estimator used in the imputation. If not provided, BayesianRidge() is used.
    :param missing_values: The values to be imputed.
    :param sample_posterior: If True, in each imputation round the intermediate estimations are samples from the
    (Gaussian) predictive posterior of the estimator. In this case estimator must support return_std in its predict
    method.
    :param max_iter: Maximum number of imputation rounds to perform. If sample_posterior=False, a stopping criterion is
    applied: max(abs(X_t - X_{t-1}))/max(abs(X[known_vals])) < tol, where X_t is X at iteration t.
    :param tol: Tolerance used in the stopping criterion.
    :param n_nearest_features: Number of features to use to estimate the missing values of a column. Features are drawn
    with a probability proportional to their absolute correlation coefficient with the column to be imputed (calculated
    after the initial imputation). If None, all features are used.
    :param initial_strategy: Strategy to be used in the initial imputation, which corresponds to the 'strategy'
    parameter of SimpleImputer.
    :param imputation_order: The order in which the features will be imputed. Possible values:
    - 'ascending': From features with fewest missing values to most
    - 'descending': From features with most missing values to fewest
    - 'roman': Left to right
    - 'arabic': Right to left
    - 'random': A random order for each round
    :param skip_complete: If True, columns that did not have any missing values during fit will only be imputed with
    SimpleImputer during transform
    :param min_value: Minimum possible imputed value(s). Can be provided for all features as a scalar or for each
    feature separately as an iterable.
    :param max_value: Maximum possible imputed value(s). Can be provided for all features as a scalar or for each
    feature separately as an iterable.
    :param random_state: Seed for a pseudo random number generator, which can be used when selecting features for
    missing values prediction, determining the order of imputation and sampling from posterior.
    """

    def __init__(
            self,
            include: set | list | None = None,
            exclude: set | list | None = None,
            include_target: bool = False,
            auto: bool = True,
            inplace: bool = False,
            estimator=None,
            missing_values: int | float | str | np.nan | None = np.nan,
            sample_posterior: bool = False,
            max_iter: int = 10,
            tol: float = 1e-3,
            n_nearest_features: int = None,
            initial_strategy: str = "default",
            imputation_order: str = "ascending",
            skip_complete=False,
            min_value=-np.inf,
            max_value=np.inf,
            random_state=None,
    ):
        Imputer.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                         include_target=include_target, missing_values=missing_values)
        self.estimator = estimator
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.random_state = random_state

    _ImputerTriplet = namedtuple(
        "_ImputerTriplet", ["feat_idx", "neighbor_feat_idx", "estimator"]
    )

    @staticmethod
    def _validate_limit(limit, limit_type: str, n_features: int):
        """
        Is used to validate the 'min_value' and 'max_value' parameters.
        :param limit: Value(s) to be validated as a scalar or iterable.
        :param limit_type: "max" or "min", used to determine the bound
        :param n_features: Number of features to be imputed
        :return: array containing the limits for each feature
        """
        limit_bound = np.inf if limit_type == "max" else -np.inf
        limit = limit_bound if limit is None else limit
        if np.isscalar(limit):
            limit = np.full(n_features, limit)
        if not limit.shape[0] == n_features:
            raise ValueError(
                f"'{limit_type}_value' should be of "
                f"shape ({n_features},) when an array-like "
                f"is provided. Got {limit.shape}, instead."
            )
        return limit

    def _impute_one_feature(
            self,
            X_filled: np.ndarray,
            mask_missing_values: np.ndarray,
            feat_idx: int,
            neighbor_feat_idx: np.ndarray,
            estimator=None,
            fit_mode: bool = True,
    ):
        """
        Performs one round of imputation for a single feature. If sample_posterior=True, the sampling from posterior
        takes place in this function.

        :param X_filled: ndarray containing features to be imputed and used for imputation.
        :param mask_missing_values: The mask of missing values.
        :param feat_idx: The index of the feature to be imputed.
        :param neighbor_feat_idx: The indices of the columns to be used to impute the feature.
        :param estimator: The estimator used in the imputation.
        :param fit_mode: If True, the estimator is fit to the data before imputation.
        :return: ndarray with the imputed feature and the estimator used in the imputation.
        """
        if estimator is None:
            estimator = clone(self._estimator)

        missing_row_mask = mask_missing_values[:, feat_idx]
        if fit_mode:
            X_train = X_filled[:, neighbor_feat_idx][~missing_row_mask]
            y_train = X_filled[:, feat_idx][~missing_row_mask]
            estimator.fit(X_train, y_train)

        # if no missing values, don't predict
        if np.sum(missing_row_mask) == 0:
            return X_filled, estimator

        # get posterior samples if there is at least one missing value
        X_test = X_filled[:, neighbor_feat_idx][missing_row_mask]
        if self.sample_posterior:
            mus, sigmas = estimator.predict(X_test, return_std=True)
            imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
            positive_sigmas = sigmas > 0
            imputed_values[~positive_sigmas] = mus[~positive_sigmas]
            mus_too_low = mus < self._min_value[feat_idx]
            imputed_values[mus_too_low] = self._min_value[feat_idx]
            mus_too_high = mus > self._max_value[feat_idx]
            imputed_values[mus_too_high] = self._max_value[feat_idx]
            # the rest can be sampled without statistical issues
            inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
            mus = mus[inrange_mask]
            sigmas = sigmas[inrange_mask]
            a = (self._min_value[feat_idx] - mus) / sigmas
            b = (self._max_value[feat_idx] - mus) / sigmas

            truncated_normal = stats.truncnorm(a=a, b=b, loc=mus, scale=sigmas)
            imputed_values[inrange_mask] = truncated_normal.rvs(
                random_state=self.random_state_
            )
        else:
            imputed_values = estimator.predict(X_test)
            imputed_values = np.clip(
                imputed_values, self._min_value[feat_idx], self._max_value[feat_idx]
            )

        # update the feature
        X_filled[missing_row_mask, feat_idx] = imputed_values
        return X_filled, estimator

    def _get_neighbor_feat_idx(self, n_features: int, feat_idx: int, abs_corr_mat: np.ndarray):
        """
        Determines features used to predict missing values in the column 'feat_idx'. If 'self.n_nearest_features' is
        less than the total number of features, randomly draws a subsample of features using probabilities
        proportional to the features' absolute correlation with column 'feat_idx'.

        :param n_features: Number of features in the dataframe.
        :param feat_idx: The index of the feature currently being imputed.
        :param abs_corr_mat: Absolute correlation matrix with the diagonal zeroed out and each feature normalized to sum
        to 1. Can be None.
        :return: indices of the features to be used to impute the feature 'feat_idx'
        """
        if self.n_nearest_features is not None and self.n_nearest_features < n_features:
            p = abs_corr_mat[:, feat_idx]
            neighbor_feat_idx = self.random_state_.choice(
                np.arange(n_features), self.n_nearest_features, replace=False, p=p
            )
        else:
            inds_left = np.arange(feat_idx)
            inds_right = np.arange(feat_idx + 1, n_features)
            neighbor_feat_idx = np.concatenate((inds_left, inds_right))
        return neighbor_feat_idx

    def _get_abs_corr_mat(self, X_filled, tolerance: float = 1e-6):
        """
        Calculates absolute correlation matrix between features.

        :param X_filled: dataframe containing features to be imputed and used for imputation.
        :param tolerance: the value with which to replace nans in 'abs_corr_mat'.
        :return: absolute correlation matrix.
        """
        n_features = X_filled.shape[1]
        if self.n_nearest_features is None or self.n_nearest_features >= n_features:
            return None
        with np.errstate(invalid="ignore"):
            # if a feature in the neighborhood has only a single value
            # (e.g., categorical feature), the std. dev. will be null and
            # np.corrcoef will raise a warning due to a division by zero
            abs_corr_mat = np.abs(np.corrcoef(X_filled.T))
        # np.corrcoef is not defined for features with zero std
        abs_corr_mat[np.isnan(abs_corr_mat)] = tolerance
        # ensures exploration, i.e. at least some probability of sampling
        np.clip(abs_corr_mat, tolerance, None, out=abs_corr_mat)
        # features are not their own neighbors
        np.fill_diagonal(abs_corr_mat, 0)
        # needs to sum to 1 for np.random.choice sampling
        abs_corr_mat = normalize(abs_corr_mat, norm="l1", axis=0, copy=False)
        return abs_corr_mat

    def _initial_imputation(self, X):
        """
        Performs initial imputation for the su%bmitted dataframe.

        :param X: the dataframe to be imputed
        :return: the initial array containing only valid features, the imputed array, the mask matrix indicating
        missing datapoints for valid features and for all features respectively
        """
        if self.initial_imputer_ is None:
            self.initial_imputer_ = SimpleImputer(
                missing_values=self.missing_values, strategy=self.initial_strategy
            )  # Sklearn
            X_filled = self.initial_imputer_.fit_transform(X)
        else:
            X_filled = self.initial_imputer_.transform(X)
        cols_to_drop = [col for col in X_filled if X_filled[col].isna().sum() > 0]
        self.params = [param for param in self.params if param not in cols_to_drop]
        X_filled = X_filled.loc[:, self.params].to_numpy()
        X = X.loc[:, self.params].to_numpy()

        X_missing_mask = _get_dense_mask(X, self.missing_values)
        mask_missing_values = X_missing_mask.copy()
        valid_mask = np.flatnonzero(
            np.logical_not(np.isnan(self.initial_imputer_.statistics_))
        )
        Xt = X[:, valid_mask]
        mask_missing_values = mask_missing_values[:, valid_mask]

        return Xt, X_filled, mask_missing_values, X_missing_mask

    def _fit_transform(self, X):
        """
        Performs imputation on the submitted dataframe. Performs validation of inputs, initial imputation and
        iterative imputation of missing values. Also, estimators are fitted to the data that can later be used on other
        datasets.

        :param X: dataframe to be imputed
        :return: array with imputed values
        """
        self.random_state_ = getattr(self, "random_state_", check_random_state(self.random_state))

        if self.max_iter < 0:
            raise ValueError("'max_iter' should be a positive integer. Got {} instead.".format(self.max_iter))

        if self.tol < 0:
            raise ValueError("'tol' should be a non-negative float. Got {} instead.".format(self.tol))

        if self.estimator is None:
            from sklearn.linear_model import BayesianRidge

            self._estimator = BayesianRidge()
        else:
            self._estimator = clone(self.estimator)

        self.imputation_sequence_ = []

        self.initial_imputer_ = None

        self.full_nan_cols = [col for col in X.columns if X[col].isnull().sum() == X.shape[0]]
        if len(self.full_nan_cols) > 0:
            X.drop(columns=self.full_nan_cols, inplace=True)
            self.params = [param for param in self.params if param not in self.full_nan_cols]

        X, Xt, mask_missing_values, complete_mask = self._initial_imputation(X)

        if self.max_iter == 0 or np.all(mask_missing_values) or Xt.shape[1] == 1:
            self.n_iter_ = 0
            return Xt

        self._min_value = self._validate_limit(self.min_value, "min", X.shape[1])
        self._max_value = self._validate_limit(self.max_value, "max", X.shape[1])

        if not np.all(np.greater(self._max_value, self._min_value)):
            raise ValueError("One (or more) features have min_value >= max_value.")

        # order in which to impute
        ordered_idx = _get_ordered_idx(mask_missing_values, self.imputation_order, self.skip_complete,
                                       self.random_state_)
        self.n_features_with_missing_ = len(ordered_idx)

        abs_corr_mat = self._get_abs_corr_mat(Xt)

        n_samples, n_features = Xt.shape
        if not self.sample_posterior:
            Xt_previous = Xt.copy()
            normalized_tol = self.tol * np.max(np.abs(X[~mask_missing_values]))
        for self.n_iter_ in range(1, self.max_iter + 1):
            if self.imputation_order == "random":
                ordered_idx = _get_ordered_idx(mask_missing_values, self.imputation_order, self.skip_complete,
                                               self.random_state_)

            for feat_idx in ordered_idx:
                neighbor_feat_idx = self._get_neighbor_feat_idx(
                    n_features, feat_idx, abs_corr_mat
                )
                Xt, estimator = self._impute_one_feature(
                    Xt,
                    mask_missing_values,
                    feat_idx,
                    neighbor_feat_idx,
                    estimator=None,
                    fit_mode=True,
                )
                estimator_triplet = IterativeImputer._ImputerTriplet(
                    feat_idx, neighbor_feat_idx, estimator
                )
                self.imputation_sequence_.append(estimator_triplet)

            if not self.sample_posterior:
                inf_norm = np.linalg.norm(Xt - Xt_previous, ord=np.inf, axis=None)
                if inf_norm < normalized_tol:
                    break
                Xt_previous = Xt.copy()
        Xt[~mask_missing_values] = X[~mask_missing_values]
        return Xt

    def _transform(self, X):
        """
        Performs imputation on the submitted array using estimators fitted previously in _fit_transform.
        Performs validation of inputs, initial imputation and iterative imputation of missing values.
        Columns containing only missing values are ignored and not imputed.

        :param X: dataframe to be imputed
        :return: array with imputed values
        """
        self.ignored = None
        if len(self.full_nan_cols) > 0:
            self.ignored = X.loc[:, self.full_nan_cols]
            SimpleImputer(strategy='constant', missing_values=self.missing_values,
                          inplace=True).fit_transform(self.ignored)
            X.drop(columns=self.full_nan_cols, inplace=True)
        X, Xt, mask_missing_values, complete_mask = self._initial_imputation(X)

        if self.n_iter_ == 0 or np.all(mask_missing_values):
            return Xt

        imputations_per_round = len(self.imputation_sequence_) // self.n_iter_
        i_rnd = 0
        for it, estimator_triplet in enumerate(self.imputation_sequence_):
            Xt, _ = self._impute_one_feature(
                Xt,
                mask_missing_values,
                estimator_triplet.feat_idx,
                estimator_triplet.neighbor_feat_idx,
                estimator=estimator_triplet.estimator,
                fit_mode=False,
            )
            if not (it + 1) % imputations_per_round:
                i_rnd += 1

        Xt[~mask_missing_values] = X[~mask_missing_values]
        return Xt

    def fit_transform(self, df):
        """
        Selects columns and calls _fit_transform, which fits estimators to the data, performs imputation and returns a
        numpy array, which is used to impute the original dataframe.

        :param df: dataframe to be imputed
        :return: dataframe with imputed values
        """
        self.params = [col for col in df if col in self._select_numeric_columns(df)]
        if not self.inplace:
            df = df.copy(deep=True)
        X = df.loc[:, self.params]
        dtypes = X.dtypes
        with df.pipeline:
            Xt = self._fit_transform(X)
        self.fitted = True
        self.reset_columns(df, Xt, dtypes)
        if len(self.full_nan_cols) > 0:
            df.loc[:, self.full_nan_cols] = 0
        df.pipeline.add(self)
        return df

    def fit(self, df):
        """
        Calls fit_transform, which fits estimators to the data and performs imputation.

        :param df: dataframe to be imputed
        :return: self
        """
        self.fit_transform(df)
        return self

    def transform(self, df):
        """
        Selects columns and calls _transform, which performs imputation and returns a numpy array, which is used to
        impute the original dataframe.

        :param df: dataframe to be imputed
        :return: dataframe with imputed values
        """
        self.validate_fit()
        if not self.inplace:
            df = df.copy(deep=True)
        X = df.loc[:, self.params]
        dtypes = X.dtypes
        with df.pipeline:
            Xt = self._transform(X)
        self.reset_columns(df, Xt, dtypes)
        df.pipeline.add(self)
        if self.ignored is not None:
            df.loc[:, self.ignored.columns] = self.ignored
        return df

    def equals(self, other):
        return Imputer.equals(self, other) \
               and self.estimator == other.estimator \
               and self.sample_posterior == other.sample_posterior \
               and self.max_iter == other.max_iter \
               and self.tol == other.tol \
               and self.n_nearest_features == other.n_nearest_features \
               and self.initial_strategy == other.initial_strategy \
               and self.imputation_order == other.imputation_order \
               and self.skip_complete == other.skip_complete \
               and self.random_state == other.random_state \
               and self.min_value == other.min_value \
               and self.max_value == other.max_value


"""
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """


class KNNImputer(Imputer):
    """
    Performs imputation of missing values based on k-Nearest Neighbors. The distance between two samples is by default
    computed based on the features that are not missing in both samples. The imputed values are weighted averages of
    the neighbors' values. The weights are determined by the weights parameter and based on the distance between the
    samples.

    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be imputed too
    :param auto: ignored for SimpleImputer, is provided for consistency
    :param inplace: if True, the transform will be completed inplace
    :param missing_values: indicates the missing value in the dataframe that will be replaced.
    :param n_neighbors: number of neighbors to use for imputation
    :param weights: determines the weights used in the imputation. Accepts following values:
    - 'uniform': uniform weights. All points in each neighborhood are weighted equally.
    - 'distance': weight points by the inverse of their distance.
    - callable: a function that takes an array of distances and returns weights.
    :param metric: distance metric for searching neighbors. Possible values:
    - 'nan_euclidean': Euclidean distance, nans are ignored.
    - callable: a user-defined function that takes two arrays are returns a scalar distance value.
    """

    def __init__(
            self,
            include: set | list | None = None,
            exclude: set | list | None = None,
            include_target: bool = False,
            auto: bool = True,
            inplace=False,
            missing_values: int | float | str | np.nan | None = np.nan,
            n_neighbors: int = 5,
            weights: {'uniform', 'distance'} | callable = "uniform",
            metric="nan_euclidean",
            **kwargs
    ):

        Imputer.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                         include_target=include_target, missing_values=missing_values)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        """
        Calculates imputed values for a single column.

        :param dist_pot_donors: Matrix of distances between receivers and potential donors
        :param n_neighbors: Number of neighbors to consider
        :param fit_X_col: Values of potential donors that are used for imputation
        :param mask_fit_X_col: Missing mask for fit_X_col
        :return: imputed values for a single column
        """
        # Get donors
        donors_idx = np.argpartition(dist_pot_donors, n_neighbors - 1, axis=1)[
                     :, :n_neighbors
                     ]

        # Get weight matrix from distance matrix
        donors_dist = dist_pot_donors[
            np.arange(donors_idx.shape[0])[:, None], donors_idx
        ]

        weight_matrix = _get_weights(donors_dist, self.weights)

        # fill nans with zeros
        if weight_matrix is not None:
            weight_matrix[np.isnan(weight_matrix)] = 0.0

        # Retrieve donor values and calculate kNN average
        donors = fit_X_col.take(donors_idx)
        donors_mask = mask_fit_X_col.take(donors_idx)
        donors = np.ma.array(donors, mask=donors_mask)

        return np.ma.average(donors, axis=1, weights=weight_matrix).data

    def _fit(self, X):
        """
        Validates inputs and calculates _valid_mask parameter, which determines features that contain some non-missing
        values.

        :param X: array to be imputed
        :return: self
        """
        if self.metric not in ["nan_euclidean"] and not callable(self.metric):
            raise ValueError("The selected metric does not support NaN values")
        if self.n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0. Got {}".format(self.n_neighbors))
        if self.weights not in (None, "uniform", "distance") and not callable(self.weights):
            raise ValueError("weights not recognized: should be 'uniform', 'distance', or a callable function")
        self._fit_X = X
        self._mask_fit_X = _get_dense_mask(self._fit_X, self.missing_values)
        self._valid_mask = ~np.all(self._mask_fit_X, axis=0)

        return self

    def _transform(self, X):
        """
        Imputes values in X.

        :param X: array to be imputed
        :return: array with imputed values
        """
        mask = _get_dense_mask(X, self.missing_values)
        mask_fit_X = self._mask_fit_X
        valid_mask = self._valid_mask

        # Return the original array if there are no missing values
        if not np.any(mask):
            return X[:, valid_mask]

        row_missing_idx = np.flatnonzero(mask.any(axis=1))

        non_missing_fix_X = np.logical_not(mask_fit_X)

        # Maps from indices from X to indices in dist matrix
        dist_idx_map = np.zeros(X.shape[0], dtype=int)
        dist_idx_map[row_missing_idx] = np.arange(row_missing_idx.shape[0])

        def process_chunk(dist_chunk, start):
            row_missing_chunk = row_missing_idx[start: start + len(dist_chunk)]

            # Find and impute missing by column
            for col in range(X.shape[1]):
                if not valid_mask[col]:
                    # column was all missing during training
                    continue

                col_mask = mask[row_missing_chunk, col]
                if not np.any(col_mask):
                    # column has no missing values
                    continue

                (potential_donors_idx,) = np.nonzero(non_missing_fix_X[:, col])

                # receivers_idx are indices in X
                receivers_idx = row_missing_chunk[np.flatnonzero(col_mask)]

                # distances for samples that needed imputation for column
                dist_subset = dist_chunk[dist_idx_map[receivers_idx] - start][
                              :, potential_donors_idx
                              ]

                # receivers with all nan distances impute with mean
                all_nan_dist_mask = np.isnan(dist_subset).all(axis=1)
                all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]

                if all_nan_receivers_idx.size:
                    col_mean = np.ma.array(
                        self._fit_X[:, col], mask=mask_fit_X[:, col]
                    ).mean()
                    X[all_nan_receivers_idx, col] = col_mean

                    if len(all_nan_receivers_idx) == len(receivers_idx):
                        # all receivers imputed with mean
                        continue

                    # receivers with at least one defined distance
                    receivers_idx = receivers_idx[~all_nan_dist_mask]
                    dist_subset = dist_chunk[dist_idx_map[receivers_idx] - start][
                                  :, potential_donors_idx
                                  ]

                n_neighbors = min(self.n_neighbors, len(potential_donors_idx))
                value = self._calc_impute(
                    dist_subset,
                    n_neighbors,
                    self._fit_X[potential_donors_idx, col],
                    mask_fit_X[potential_donors_idx, col],
                )
                X[receivers_idx, col] = value

        # process in fixed-memory chunks
        gen = pairwise_distances_chunked(
            X[row_missing_idx, :],
            self._fit_X,
            metric=self.metric,
            missing_values=self.missing_values,
            reduce_func=process_chunk,
        )
        for chunk in gen:
            # process_chunk modifies X in place. No return value.
            pass

        return X[:, valid_mask]

    def fit(self, df):
        """
        Performs column selection and calls _fit.

        :param df: dataframe to be imputed
        :return: self
        """
        self.params = [col for col in df if col in self._select_numeric_columns(df)]
        self.full_nan_cols = [col for col in df.columns if df[col].isnull().sum() == df.shape[0]]
        with df.pipeline:
            self._fit(df.loc[:, self.params].to_numpy())
        self.fitted = True
        return self

    def transform(self, df):
        """
        Selects columns and calls _transform, which performs imputation and returns a numpy array, which is used to
        impute the original dataframe. The columns that contain all nan values during fit are imputed with value 0.

        :param df: dataframe to be imputed
        :return: dataframe with imputed values
        """
        dtypes = df.dtypes[self.params]
        self.validate_fit()
        if not self.inplace:
            maybe_df = df.copy(deep=True)
        else:
            maybe_df = df
        with maybe_df.pipeline:
            Xt = self._transform(maybe_df.loc[:, self.params].to_numpy())
        self.params = [param for param in self.params if param not in self.full_nan_cols]
        maybe_df.loc[:, self.params] = Xt
        for col in self.full_nan_cols:
            maybe_df[col][maybe_df[col].isna()] = 0
        self.reset_columns(maybe_df, Xt, dtypes)
        maybe_df.pipeline.add(self)
        if not self.inplace:
            return maybe_df

    def equals(self, other):
        return Imputer.equals(self, other) \
               and self.metric == other.metric \
               and self.n_neighbors == other.n_neighbors \
               and self.weights == other.weights
