"""
The underlying code was partially inspired by the scipy
(https://github.com/scipy/scipy) trim / winsorize functionality.
No scipy code was utilized during the creation of this class.
Nevertheless, the usage of the trim / winsorize ideas from scipy
would be allowed under scipy's BSD license:

Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Footer

Please also refer to this packages LICENSES folder.

"""

from __future__ import annotations

import numbers
from abc import abstractmethod

import numpy as np

from ..core.pipeline.lock import Lock
from .generic import _BaseProcessor


class _BaseTrimmer(_BaseProcessor):
    """
    Base class for trimmer / winsorizer classes.

    :param include: if provided, the dataframe will be trimmed / winsorized only
    according to these columns
    :param exclude: if provided, the transform will not be trimmed / winsorized
    according to these columns
    :param include_target: if True, the dataframe will be trimmed / winsorized
    according to target column as well
    :param auto: if True, the transform will be attempted only for numeric columns
    :param inplace: if True, the transform will be completed inplace
    :param limits: if relative is True, the top / bottom percentiles to be excluded
    / winsorized for each column. If relative is False, all feature values below first /
    above second threshold will be trimmed / winsorized
    :param inclusive: If True, the values that correspond to the exact limits will
    not be trimmed. If False, they will be trimmed / winsorized
    :param relative: if True, limits will be interpreted as percentiles. If False,
    the limits will be interpreted as absolute values
    """
    def __init__(self, inplace: bool = False, auto: bool = True, include: set | list | None = None,
                 exclude: set | list | None = None, include_target: bool = False, limits: tuple | dict = (0.01, 0.01),
                 inclusive: tuple = (True, True), relative: bool = True):
        _BaseProcessor.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                                include_target=include_target)
        self.inclusive = inclusive
        self.limits = limits
        self.relative = relative
        self.params = None

    def fit(self, df):
        """
        Fits the trimmer / winsorizer. Calculates the respective quantiles
        for each columns if relative is True

        :param df: dataframe to be fitted
        """
        with Lock.lock():
            columns = self.select_columns(df)
            self._validate_limits(columns)

            self.params = {}

            if not self.relative:
                # shortcut - if the limits are absolute, we don't need to look through columns
                if isinstance(self.limits, dict):
                    self.params = self.limits
                else:
                    self.params = {col: self.limits for col in columns}
                self.fitted = True
                return
            self._fit(df, columns)
        self.fitted = True

    def select_columns(self, df):
        """
        Selects (if auto - numeric) columns

        :param df: dataframe to be fitted / transformed
        :return: selected columns
        """
        return self._select_numeric_columns(df)

    def _validate_limits(self, columns: set):
        """
        The function that validates the provided limits

        :param columns: columns to be fitted / transformed
        """
        if isinstance(self.limits, dict):
            keys = set(self.limits.keys())

            limits_not_in_columns = keys - columns
            if limits_not_in_columns:
                raise ValueError(f"Provided limits have keys not selected in columns: {limits_not_in_columns}")

            columns_not_in_limits = columns - keys
            if columns_not_in_limits:
                raise ValueError(
                    f"Selected columns contain keys not present in provided limits: {columns_not_in_limits}")

    def _fit(self, df, columns: set):
        """
        The core fit function. Calculates the absolute values for
        later transformation

        :param df: dataframe to be fitted
        :param columns: columns to be transformed
        """
        for col in columns:
            nd = df[col].to_numpy()

            if self.relative:
                if isinstance(self.limits, dict):
                    self.params[col] = np.nanpercentile(nd, self._get_percentages(*self.limits[col]), axis=0)
                else:
                    self.params[col] = np.nanpercentile(nd, self._get_percentages(*self.limits), axis=0)
            else:
                if isinstance(self.limits, dict):
                    self.params[col] = self.limits[col]

    @staticmethod
    def _get_percentages(low: numbers.Number, high: numbers.Number):
        return low * 100, (1 - high) * 100

    @abstractmethod
    def transform(self, df):
        pass

    def equals(self, other):
        return _BaseProcessor.equals(self, other) and \
               self.inclusive == other.inclusive and \
               self.limits == other.limits and \
               self.relative == other.relative


class Trimmer(_BaseTrimmer):

    def transform(self, df):
        """
        Trims the data according to fitted limits

        :param df: dataframe to be trimmed
        :return: transformed dataframe if not inplace
        """
        maybe_df = df if self.inplace else df.copy(deep=True)
        maybe_pipeline = maybe_df.pipeline
        maybe_target = maybe_df.target
        with Lock.lock():
            if not self.fitted:
                raise RuntimeError(
                    'The ' + self.__class__.__name__ + ' was not fitted.\nPlease '
                                                       'fit it first before attempting the transformation.')

            if not self.inplace:
                maybe_df = df.copy(deep=True)
            else:
                maybe_df = df

            if self.params:
                mask = None

                for col in self.params:
                    low, high = self.params[col]  # just for readability
                    nd = maybe_df[col].to_numpy()
                    if mask is None:
                        mask = (((nd >= low if self.inclusive[0] else nd > low) & (
                            nd <= high if self.inclusive[1] else nd < high))) | np.isnan(nd)
                    else:
                        mask &= (((nd >= low if self.inclusive[0] else nd > low) & (
                            nd <= high if self.inclusive[1] else nd < high)) | np.isnan(nd))

                maybe_df.__init__(maybe_df[mask], target=maybe_df.target, pipeline=maybe_df.pipeline.add(self))
        maybe_df.pipeline = maybe_pipeline.add(self)
        maybe_df.target = maybe_target

        if not self.inplace:
            return maybe_df


class Winsorizer(_BaseTrimmer):

    def transform(self, df):
        """
        Winsorizes the data according to fitted limits

        :param df: dataframe to be winsorized
        :return: transformed dataframe if not inplace
        """
        maybe_df = df if self.inplace else df.copy(deep=True)
        maybe_pipeline = maybe_df.pipeline
        maybe_target = maybe_df.target
        with Lock.lock():
            if not self.fitted:
                raise RuntimeError(
                    'The ' + self.__class__.__name__ + ' was not fitted.\nPlease '
                                                       'fit it first before attempting the transformation.')

            for col in self.params:
                low, high = self.params[col]  # just for readability
                nd = maybe_df[col].to_numpy()
                eps = np.finfo(nd.dtype).eps if np.issubdtype(nd.dtype, np.inexact) else 1
                if self.inclusive[0]:
                    nd[nd < low] = low
                else:
                    nd[nd <= low] = low + eps
                if self.inclusive[1]:
                    nd[nd > high] = high
                else:
                    nd[nd >= high] = high - eps

        maybe_df.pipeline = maybe_pipeline.add(self)
        maybe_df.target = maybe_target
        if not self.inplace:
            return maybe_df
