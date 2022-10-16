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

import warnings
from abc import abstractmethod

import numpy as np
from scipy import stats  # provided through sklearn

from ..core.pipeline.lock import Lock
from .generic import _BaseProcessor
from ..utils.pd_utils.pd_utils import find_stack_level


class _BaseScaler(_BaseProcessor):
    """
    Base class for dataframe scaling

    """
    def __init__(self,
                 include: set | list | None = None,
                 exclude: set | list | None = None,
                 include_target: bool = False,
                 auto: bool = True,
                 inplace: bool = False,
                 fast_transform: bool = False):
        """
        :param include: if provided, the transform will be applied only to these columns
        :param exclude: if provided, the transform will not be applied to these columns
        :param include_target: if True, the target column will be scaled too
        :param inplace: if True, the transform will be completed inplace
        :param auto: if True, the transform will only be attempted for numeric columns
        :param fast_transform: attempts to transform the data as-is -
        in particular, no auto-detection of columns to be transformed will be
        attempted. 'include' or 'exclude' arguments cannot be provided together
        with fast_transform. If a target is provided and 'include_target' is set to
        False, the data won't be transformed 'as-is'. Default False.
        """
        _BaseProcessor.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                                include_target=include_target)
        self.blockwise = self.include is None and self.exclude is None
        if fast_transform:
            if not self.blockwise:
                raise ValueError('cannot provide include or exclude when using fast transform')
            if auto:
                warnings.warn(
                    "Auto and fast transformed shouldn't be provided together, "
                    "the automatic selection of columns to be transformed won't "
                    "be attempted",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
        self.fast_transform = fast_transform

    def fit(self, df):
        """
        Base fit function. Fits the data blockwise (i.e., by
        scaling the whole numpy array at the same time) if possible,
        otherwise scales each column separately.

        :param df: dataframe to be fitted
        """

        with Lock.lock():
            self.blockwise = self.blockwise and (
                    not df.target or self.include_target) and _BaseScaler.check_numeric_type_consistency(df)
            with df.pipeline:
                if self.blockwise:
                    self._fit_blockwise(df)
                else:
                    self._fit_columnwise(df)

            self.fitted = True

    def transform(self, df):
        """
        Base transform function. If not fast_transform,
        the transformation will be completed columnwise.

        :param df: dataframe to be transformed
        :return: transformed dataframe if not inplace
        """

        self.validate_fit()
        maybe_df = df if self.inplace else df.copy(deep=True)
        maybe_pipeline = maybe_df.pipeline
        maybe_target = maybe_df.target
        with Lock.lock():
            with maybe_df.pipeline:
                self._transform_blockwise(maybe_df) if self.blockwise and self.fast_transform \
                    else self._transform_columnwise(maybe_df)

        maybe_df.pipeline = maybe_pipeline.add(self)
        maybe_df.target = maybe_target
        if not self.inplace:
            return maybe_df

    def select_columns(self, df):
        """
        Selects columns according to include & exclude.
        If auto, returns only numeric columns.

        :param df: dataframe to be fitted
        :return: selected numeric columns
        """

        return self._select_numeric_columns(df)

    def __eq__(self, other):
        return _BaseProcessor.__eq__(self, other) \
               and self.blockwise == other.blockwise \
               and self.fast_transform == other.fast_transform

    def _add_maybe_blockwise(self, attribute: np.ndarray, columns: list, attribute_name: str):
        """
        Technical function to add fitted parameters.
        If fast_transform, adds parameters as a numpy array,
        otherwise adds a dictionary of attributes

        :param attribute: fitted parameters in a numpy array
        :param columns: list of columns for which the params
        should be added
        :param attribute_name: the name of the attribute to
        be added
        """
        object.__setattr__(self, attribute_name,
                           attribute if self.fast_transform else {col: attribute[i] for i, col in enumerate(columns)})

    @abstractmethod
    def _fit_blockwise(self, df):
        """
        Abstract method to fit the scaler to a dataframe
        as one numpy array

        :param df: dataframe to be fitted
        """
        pass

    @abstractmethod
    def _fit_columnwise(self, df):
        """
        Abstract method to fit the scaler one column at a time

        :param df: dataframe to be fitted
        """
        pass

    @abstractmethod
    def _transform_columnwise(self, df):
        """
        Abstract method to transform the dataframe one column
        at a time

        :param df: dataframe to be transformed
        :return: transformed dataframe
        """
        pass

    @abstractmethod
    def _transform_blockwise(self, df):
        """
        Abstract method to transform the dataframe as one
        numpy array

        :param df: dataframe to be transformed
        :return: transformed dataframe
        """
        pass


class StandardScaler(_BaseScaler):
    """
    Standardizes the data in dataframe's columns. Given a column x, calculates

    x = (x - x.mean()) / x.std()

    If the data in the column is almost constant, the column won't be rescaled.

    Supports the following arguments:
    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be scaled too
    :param inplace: if True, the transform will be completed inplace
    :param auto: if True, the transform will only be attempted for numeric columns
    :param fast_transform: attempts to transform the data as-is -
    in particular, no auto-detection of columns to be transformed will be
    attempted. 'include' or 'exclude' arguments cannot be provided together
    with fast_transform. If a target is provided and 'include_target' is set to
    False, the data won't be transformed 'as-is'. Default False.
    :param with_mean: if the mean should be subtracted from data
    :param with_std: if the data should be scaled with its standard deviation
    """

    def __init__(self,
                 include: set | list | None = None,
                 exclude: set | list | None = None,
                 include_target: bool = False,
                 auto: bool = True,
                 inplace: bool = False,
                 fast_transform: bool = False,
                 with_mean: bool = True,
                 with_std: bool = True):
        _BaseScaler.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                             include_target=include_target, fast_transform=fast_transform)
        self.with_mean = with_mean
        self.with_std = with_std
        self.means = None
        self.stds = None

    def _fit_blockwise(self, df):
        """
        Fits the scaler to a dataframe as one numpy array

        :param df: dataframe to be fitted
        """
        if self.auto and not self.fast_transform:
            df = df.select_dtypes(include=np.number)
        nd = df.to_numpy()
        if self.with_mean:
            means = np.nanmean(nd, axis=0)
            self._add_maybe_blockwise(means, df.columns, 'means')
        if self.with_std:
            stds = np.nanstd(nd, axis=0)
            self._add_maybe_blockwise(stds, df.columns, 'stds')

    def _fit_columnwise(self, df):
        """
        Fits the scaler to a dataframe one column at a time

        :param df: dataframe to be fitted
        """
        cols = self.select_columns(df)
        self.means = {} if self.with_mean else None
        self.stds = {} if self.with_std else None
        for col in cols:
            nd = df[col].to_numpy()
            if self.with_mean:
                self.means[col] = np.nanmean(nd)
            if self.with_std:
                self.stds[col] = np.nanstd(nd)

    def _transform_columnwise(self, df):
        """
        Scales the data in the dataframe one column
        at a time

        :param df: dataframe to be transformed
        :return: transformed dataframe if not inplace
        """
        if self.with_mean:
            if self.with_std:
                for col in self.means:
                    nd = df[col].to_numpy()
                    current_std = self.stds[col]
                    if current_std > 10 * np.finfo(current_std.dtype).eps:  # handle near-constant data
                        df[col] = (nd - self.means[col]) / current_std
            else:
                for col in self.means:
                    nd = df[col].to_numpy()
                    df[col] = nd - self.means[col]
        elif self.with_std:
            for col in self.stds:
                nd = df[col].to_numpy()
                current_std = self.stds[col]
                if current_std > 10 * np.finfo(current_std.dtype).eps:  # handle near-constant data
                    df[col] = nd / current_std

    def _transform_blockwise(self, df):
        """
        Scales the data in the dataframe as one numpy array.
        Is completed only if self.fast_transform is True.

        :param df: dataframe to be transformed
        :return: transformed dataframe if not inplace
        """
        nd = df.to_numpy()
        stds = self.stds
        if stds is not None:
            stds[stds <= 10 * np.finfo(stds.dtype).eps] = 1
        if self.with_mean:
            nd -= self.means
        if self.with_std:
            nd /= stds
        df.reinitialize(nd)

    def equals(self, other):
        return _BaseScaler.equals(self, other) \
               and self.with_mean == other.with_mean \
               and self.with_std == other.with_std


class MinMaxScaler(_BaseScaler):
    """
    Scales the data in dataframe's columns so that it is greater than a and
    smaller than b, where a, b are the first and the second element in the
    feature range. Default values are 0 for a and 1 for b.

    Supports the following arguments:
    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be scaled too
    :param inplace: if True, the transform will be completed inplace
    :param auto: if True, the transform will only be attempted for numeric columns
    :param fast_transform: attempts to transform the data as-is -
    in particular, no auto-detection of columns to be transformed will be
    attempted. 'include' or 'exclude' arguments cannot be provided together
    with fast_transform. If a target is provided and 'include_target' is set to
    False, the data won't be transformed 'as-is'. Default False.
    :param feature_range: provides the range to which the data will be scaled
    :param clip: if the data larger than upper bound of range and lower than
    lower bound of range should be clipped
    """
    def __init__(self,
                 include: set | list | None = None,
                 exclude: set | list | None = None,
                 include_target: bool = False,
                 auto: bool = True,
                 inplace: bool = False,
                 fast_transform: bool = False,
                 feature_range=(0, 1),
                 clip: bool = False):
        _BaseScaler.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                             include_target=include_target, fast_transform=fast_transform)
        self.feature_range = feature_range
        self.clip = clip

    def _fit_blockwise(self, df):
        """
        Fits the scaler to a dataframe as one numpy array

        :param df: dataframe to be fitted
        """
        if self.auto and not self.fast_transform:
            df = df.select_dtypes(include=np.number)
        data_min = df.min(axis=0)
        data_range = df.max(axis=0) - data_min
        range_ = self.feature_range[1] - self.feature_range[0]
        data_range[data_range < 10 * np.finfo(data_range.dtype).eps] = range_
        data_diff = range_ / data_range
        if self.fast_transform:
            self.params = np.stack((data_diff, data_min))
        else:
            self.params = {col: (data_diff[i], data_min[i]) for i, col in enumerate(df.columns)}

    def _fit_columnwise(self, df):
        """
        Fits the scaler to a dataframe one column at a time

        :param df: dataframe to be fitted
        """
        cols = self.select_columns(df)
        self.params = {}
        range_ = self.feature_range[1] - self.feature_range[0]
        for col in cols:
            sel = df[col]
            current_min = sel.min(axis=0)
            current_range = sel.max(axis=0) - current_min
            if current_range < 10 * np.finfo(current_range.dtype).eps:
                self.params[col] = (1, current_min)
            else:
                self.params[col] = (range_ / current_range, current_min)

    def _transform_columnwise(self, df):
        """
        Scales the data in the dataframe one column
        at a time

        :param df: dataframe to be transformed
        :return: transformed dataframe if not inplace
        """
        min_ = self.feature_range[0]
        max_ = self.feature_range[1]

        for col in self.params:
            current_params = self.params[col]
            nd = (df[col].to_numpy() - current_params[1]) * current_params[0] + min_
            if self.clip:
                nd[nd > max_] = max_
                nd[nd < min_] = min_
            df[col] = nd

    def _transform_blockwise(self, df):
        """
        Scales the data in the dataframe as one numpy array.
        Is completed only if self.fast_transform is True.

        :param df: dataframe to be transformed
        :return: transformed dataframe if not inplace
        """
        min_ = self.feature_range[0]
        max_ = self.feature_range[1]

        nd = df.to_numpy()
        nd = (nd - self.params[1]) * self.params[0] + min_
        if self.clip:
            nd[nd > max_] = max_
            nd[nd < min_] = min_
        df.reinitialize(nd)

    def equals(self, other):
        return _BaseScaler.equals(self, other) \
               and self.feature_range == other.feature_range \
               and self.clip == other.clip


class MaxAbsScaler(_BaseScaler):
    """
    Scales the data in dataframe's columns so that the maximum absolute value
    in each column is equal to 1.

    Supports the following arguments:
    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be scaled too
    :param inplace: if True, the transform will be completed inplace
    :param auto: if True, the transform will only be attempted for numeric columns
    :param fast_transform: attempts to transform the data as-is -
    in particular, no auto-detection of columns to be transformed will be
    attempted. 'include' or 'exclude' arguments cannot be provided together
    with fast_transform. If a target is provided and 'include_target' is set to
    False, the data won't be transformed 'as-is'. Default False.
    """

    def __init__(self,
                 include: set | list | None = None,
                 exclude: set | list | None = None,
                 include_target: bool = False,
                 auto: bool = True,
                 inplace: bool = False,
                 fast_transform: bool = False):
        _BaseScaler.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                             include_target=include_target, fast_transform=fast_transform)
        self.params = None

    def _fit_columnwise(self, df):
        """
        Fits the scaler to a dataframe one column at a time

        :param df: dataframe to be fitted
        """
        cols = self.select_columns(df)
        self.params = {}
        for col in cols:
            sel = df[col]
            current_abs = np.maximum(np.abs(sel.max(axis=0)), np.abs(sel.min(axis=0)))
            if current_abs < 10 * np.finfo(current_abs.dtype).eps:
                self.params[col] = 1
            else:
                self.params[col] = current_abs

    def _fit_blockwise(self, df):
        """
        Fits the scaler to a dataframe as one numpy array

        :param df: dataframe to be fitted
        """
        if self.auto and not self.fast_transform:
            df = df.select_dtypes(include=np.number)
        mins = df.min(axis=0).to_numpy()
        maxs = df.max(axis=0).to_numpy()
        params = np.maximum(np.abs(mins), np.abs(maxs))
        params[params < 10 * np.finfo(params.dtype).eps] = 1
        self._add_maybe_blockwise(params, df.columns, 'params')

    def _transform_columnwise(self, df):
        """
        Scales the data in the dataframe one column
        at a time

        :param df: dataframe to be transformed
        :return: transformed dataframe if not inplace
        """
        for col in self.params:
            nd = df[col].to_numpy() / self.params[col]
            df[col] = nd

    def _transform_blockwise(self, df):
        """
        Scales the data in the dataframe as one numpy array.
        Is completed only if self.fast_transform is True.

        :param df: dataframe to be transformed
        :return: transformed dataframe if not inplace
        """
        nd = df.to_numpy() / self.params
        df.reinitialize(nd)


class RobustScaler(_BaseScaler):
    """
    Scales the data in dataframe's columns robustly. It subtracts the median and
    scales the data according to the provided quantile range.

    Supports the following arguments:
    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be scaled too
    :param inplace: if True, the transform will be completed inplace
    :param auto: if True, the transform will only be attempted for numeric columns
    :param fast_transform: attempts to transform the data as-is -
    in particular, no auto-detection of columns to be transformed will be
    attempted. 'include' or 'exclude' arguments cannot be provided together
    with fast_transform. If a target is provided and 'include_target' is set to
    False, the data won't be transformed 'as-is'. Default False.
    :param with_centering: if the median should be subtracted
    :param with_scaling: if the data should be scaled to the provided quantile range
    :param quantile_range: the quantiles to which the data should be scaled. E.g. if
    (25.0, 75.0) is provided, the data will be scaled with the difference between 75%
    and 25% quantiles.
    :param unit_variance: if the normally distributed features should be scaled to unit
    variance
    """

    def __init__(self,
                 include: set | list | None = None,
                 exclude: set | list | None = None,
                 include_target: bool = False,
                 auto: bool = True,
                 inplace: bool = False,
                 fast_transform: bool = False,
                 with_centering: bool = True,
                 with_scaling: bool = True,
                 quantile_range: tuple = (25.0, 75.0),
                 unit_variance: bool = False,
                 ):
        _BaseScaler.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                             include_target=include_target, fast_transform=fast_transform)
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        if not 0 <= self.quantile_range[0] < self.quantile_range[1] <= 100:
            raise ValueError('Provided quantile range is not valid')
        self.unit_variance = unit_variance
        self.center = None
        self.scaling = None

    def _fit_blockwise(self, df):
        """
        Fits the scaler to a dataframe as one numpy array

        :param df: dataframe to be fitted
        """
        if self.auto and not self.fast_transform:
            df = df.select_dtypes(include=np.number)
        nd = df.to_numpy()
        if self.unit_variance:
            adjust = stats.norm.ppf(self.quantile_range[1] / 100.0) - stats.norm.ppf(self.quantile_range[0] / 100.0)
        else:
            adjust = 1
        if self.with_centering:
            center = np.nanmedian(nd, axis=0)
            self._add_maybe_blockwise(center, df.columns, 'center')

        if self.with_scaling:
            percentiles = np.nanpercentile(nd, self.quantile_range, axis=0)
            scaling = (percentiles[1] - percentiles[0]) / adjust
            scaling[scaling < 10 * np.finfo(scaling.dtype).eps] = 1
            self._add_maybe_blockwise(scaling, df.columns, 'scaling')

    def _fit_columnwise(self, df):
        """
        Fits the scaler to a dataframe one column at a time

        :param df: dataframe to be fitted
        """
        cols = self.select_columns(df)
        self.scaling = {} if self.with_centering else None
        self.center = {} if self.with_scaling else None
        if self.unit_variance:
            adjust = stats.norm.ppf(self.quantile_range[1] / 100.0) - stats.norm.ppf(self.quantile_range[0] / 100.0)
        else:
            adjust = 1

        for col in cols:
            sel = df[col].to_numpy()
            if self.with_centering:
                current_center = np.nanmedian(sel)
                self.center[col] = current_center

            if self.with_scaling:
                current_scale = np.nanpercentile(sel, self.quantile_range)
                current_scale = current_scale[1] - current_scale[0]
                if current_scale < 10 * np.finfo(current_scale.dtype).eps:
                    self.scaling[col] = 1
                else:
                    self.scaling[col] = current_scale / adjust

    def _transform_blockwise(self, df):
        """
        Scales the data in the dataframe as one numpy array.
        Is completed only if self.fast_transform is True.

        :param df: dataframe to be transformed
        :return: transformed dataframe if not inplace
        """
        nd = df.to_numpy()
        if self.with_centering:
            nd -= self.center
        if self.with_scaling:
            nd /= self.scaling
        df.reinitialize(nd)

    def _transform_columnwise(self, df):
        """
        Scales the data in the dataframe one column
        at a time

        :param df: dataframe to be transformed
        :return: transformed dataframe if not inplace
        """
        if self.with_centering:
            if self.with_scaling:
                for col in self.scaling:
                    df[col] = (df[col].to_numpy() - self.center[col]) / self.scaling[col]
            else:
                for col in self.center:
                    df[col] = df[col].to_numpy() - self.center[col]
        elif self.with_scaling:
            for col in self.scaling:
                df[col] = df[col].to_numpy() / self.scaling[col]

    def equals(self, other):
        return _BaseScaler.equals(self, other) \
               and self.with_centering == other.with_centering \
               and self.with_scaling == other.with_scaling \
               and self.quantile_range == other.quantile_range \
               and self.unit_variance == other.unit_variance
