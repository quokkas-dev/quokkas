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

import math
import numbers
from abc import abstractmethod

import numpy as np
import pandas as pd
from pandas import unique
from pandas.core.dtypes.common import is_integer_dtype

from ..core.pipeline.lock import Lock
from .generic import _BaseProcessor
from ..utils.pd_utils.pd_typing import Dtype
from ..utils.sk_utils import _nandict, is_scalar_nan, check_sorted, object_diff, _unique_python


class _BaseEncoder(_BaseProcessor):
    """
    Base class for the quokkas implementations of categorical
    feature encoders. Supports the following arguments:

    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be encoded too
    :param auto: if True, the transform will only be attempted for columns that can
    be identified as categorical. Columns are identified as categorical if there are
    20 or fewer values and the share of distinct values is less than 10%
    :param inplace: if True, the transform will be completed inplace
    :param dtype: the type to which the columns will be encoded
    :param categories: the values of the categories in each column to be
    encoded as a dictionary
    :param handle_unknown: the strategy to deal with not previously en-
    countered values in the encoder
    """
    CATEGORICAL_SHARE = 0.1
    CATEGORICAL_NUMBER = 20
    ALLOWED_HANDLE_UNKNOWN = {}
    REQUIRES_SORTING = False

    def __init__(self,
                 include: set | list | None = None,
                 exclude: set | list | None = None,
                 include_target: bool = False,
                 auto: bool = True,
                 inplace: bool = False,
                 dtype: Dtype = np.float64,
                 categories: str | dict = 'auto',
                 handle_unknown: str = 'error'):
        _BaseProcessor.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                                include_target=include_target)
        self.categories = categories
        self.dtype = dtype
        if handle_unknown not in self.ALLOWED_HANDLE_UNKNOWN:
            raise ValueError(
                "received unknown handle_unknown parameter"
                f"{handle_unknown}. Expected one of:"
                f" {self.ALLOWED_HANDLE_UNKNOWN}"
            )
        self.handle_unknown = handle_unknown

    def fit_transform(self, df):
        """
        Fits the Encoder to the provided data and transforms
        the dataframe directly afterwards. Importantly, it passes
        an additional parameter 'ensure_validity' = False to transform,
        so that it does not attempt to ensure that the data was fitted
        on consistent data

        :param df: dataframe to be transformed
        :return: transformed dataframe if inplace
        """
        self.fit(df)
        return self.transform(df, ensure_validity=False)

    def fit(self, df):
        """
        Fits the encoder to the provided data

        :param df: dataframe to be fitted
        """
        with Lock.lock():
            columns = self.select_columns(df)
            self._autofit(df, columns) if self.auto else self._fit(df, columns)
            self.fitted = True

    def select_columns(self, df):
        """
        Selects columns based on include, exclude and include_target.
        Additionally, if auto, it excludes already encoded columns

        :param df: dataframe from which the columns are selected
        :return: selected columns
        """
        cols = _BaseProcessor._select_columns(self, df)

        if self.include is not None:
            return cols

        if self.auto:
            cols = df.pipeline.reduce_encoded_cols(cols)

        return cols

    def _autofit(self, df, columns: list):
        """
        Fits the encoder to the provided data. Is only called
        if auto is True. In particular, it fits only those columns
        that can be determined as categorical

        :param df: dataframe to be fitted
        :param columns: list of columns that should be fitted
        """

        border = min(OrdinalEncoder.CATEGORICAL_NUMBER, OrdinalEncoder.CATEGORICAL_SHARE * df.shape[0])
        self.params = {}
        for col in columns:
            if self._save_categories_if_provided(col, df):
                continue
            uniques = unique(df[col])
            if len(uniques) <= border:
                self._save_uniques(col, uniques)

    def _fit(self, df, columns: list):
        """
        Fits the encoder to the provided data. Is only called
        if auto is False.

        :param df: dataframe to be fitted
        :param columns: list of columns that should be fitted
        """

        self.params = {}
        for col in columns:
            if self._save_categories_if_provided(col, df):
                continue
            self._save_uniques(col, unique(df[col]))

    def _save_categories_if_provided(self, col, df) -> bool:
        """
        Saves the categories if they are provided by user

        :param col: column for which the categories should
        be saved
        :param df: dataframe to be fitted
        :return: bool if the categories provided by user
        were used
        """

        if self.categories != 'auto':
            if col in self.categories:
                if self.REQUIRES_SORTING:
                    check_sorted(self.categories[col], df[col].dtype)
                self.params[col] = self.categories[col]
                return True
        return False

    def equals(self, other):
        return _BaseProcessor.equals(self, other) \
               and self.dtype == other.dtype \
               and self.handle_unknown == other.handle_unknown

    @staticmethod
    def _reduce_nan(diff: np.ndarray, params: np.ndarray, infrequents: np.ndarray | None = None):
        """
        Static method that reduces nans in the difference between
        fitted unique values and the actual unique values in the data.

        Can be applied only to not "OUS"-data

        :param diff: difference array between actual unique values
        and fitted unique values
        :param params: fitted unique values
        :param infrequents: values to be mapped as infrequent
        :return: reduced difference array
        """
        if diff.size:
            if np.isnan(diff).any() and np.isnan(params).any():
                diff = diff[~np.isnan(diff)]
        if infrequents is not None:
            diff = diff[~np.isin(diff, infrequents)]
        return diff

    @abstractmethod
    def transform(self, df, ensure_validity=True):
        """
        Encodes the provided data

        :param df: dataframe to be transformed
        :return: transformed dataframe if inplace
        """
        pass

    @abstractmethod
    def _save_uniques(self, col, uniques):
        """
        Saves unique feature values for a column

        :param col: column for which the data is saved
        :param uniques: the unique values
        """
        pass


class OrdinalEncoder(_BaseEncoder):
    """
    Encodes categorical columns as ordinal values.

    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be encoded too
    :param auto: if True, the transform will only be attempted for columns that can
    be identified as categorical. Columns are identified as categorical if there are
    20 or fewer values and the share of distinct values is less than 10%
    :param inplace: if True, the transform will be completed inplace
    :param dtype: the type to which the columns will be encoded
    :param categories: the values of the categories in each column to be
    encoded as a dictionary
    :param handle_unknown: the strategy to deal with not previously en-
    countered values in the encoder. Can be 'error' or 'use_encoded_value'. If
    'error', throws an error if an unknown category is encountered. If
    'use_encoded_value', replaces the new with unknown_value
    :param unknown_value: if 'use_encoded_value', this value is used to replace
    the values that were not encountered during fit
    :param encoded_missing_value: a value to be used when encoding nans. If None,
    the nans are encoded like any other value
    """

    ALLOWED_HANDLE_UNKNOWN = {'error', 'use_encoded_value'}
    REQUIRES_SORTING = True

    def __init__(self,
                 include: set | list | None = None,
                 exclude: set | list | None = None,
                 include_target: bool = False,
                 auto: bool = True,
                 inplace: bool = False,
                 dtype: Dtype = np.float64,
                 categories: str | dict = 'auto',
                 handle_unknown: str = 'error',
                 unknown_value=None,
                 encoded_missing_value=None):
        _BaseEncoder.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                              include_target=include_target, categories=categories, dtype=dtype,
                              handle_unknown=handle_unknown)

        if handle_unknown == 'use_encoded_value':
            if not isinstance(unknown_value, numbers.Integral) and not math.isnan(unknown_value):
                raise ValueError("unknown_value should be nan or an integer")

        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value

        self.blockwise = self.include is None and self.exclude is None
        self.params = None

    def transform(self, df, ensure_validity=True):
        """
        Transforms the provided dataframe.

        :param df: dataframe to be transformed
        :param ensure_validity: if True, will attempt to find values
        that were not encountered during training. Should be set to
        False only if the encoder was fitted to the same data it is
        transforming
        :return: transformed dataframe
        """
        self.validate_fit()
        maybe_df = df if self.inplace else df.copy(deep=True)
        maybe_pipeline = maybe_df.pipeline
        maybe_target = maybe_df.target

        with Lock.lock():
            for col in self.params:
                nd = maybe_df[col].to_numpy()
                uniques = unique(nd)
                mask_unknown = None
                nanmask = None
                if nd.dtype.kind in "OUS":
                    diff = object_diff(uniques, self.params[col]) if ensure_validity else np.empty([0], dtype=nd.dtype)
                    if diff and self.handle_unknown == 'error':
                        raise RuntimeError(f'OrdinalEncoder.transform failed - '
                                           f'unseen category in the column {str(col)}')

                    if self.encoded_missing_value is not None:
                        dct = {val: i if not is_scalar_nan(val) else self.encoded_missing_value for i, val in
                               enumerate(self.params[col])}
                    else:
                        dct = {val: i for i, val in enumerate(self.params[col])}
                    if diff:
                        for key in diff:
                            dct[key] = self.unknown_value
                    table = _nandict(dct)
                    nd = np.array([table[v] for v in nd])
                else:
                    diff = np.setdiff1d(uniques, self.params[col], assume_unique=True) if ensure_validity else np.empty(
                        [0], dtype=nd.dtype)
                    diff = OrdinalEncoder._reduce_nan(diff, self.params[col])
                    if self.encoded_missing_value is not None:
                        nanmask = np.isnan(nd)
                    if diff.size:
                        if self.handle_unknown == 'error':
                            raise RuntimeError(
                                'OrdinalEncoder.transform failed - unseen category in the column ' + str(col))
                        else:
                            mask_unknown = np.isin(nd, diff)
                            nd[mask_unknown] = self.params[col][0]
                    nd = np.searchsorted(self.params[col], nd)

                    if mask_unknown is not None:
                        nd[mask_unknown] = self.unknown_value
                if nanmask is not None:
                    nd = nd.astype(self.dtype)
                    nd[nanmask] = self.encoded_missing_value
                    maybe_df[col] = nd
                else:
                    maybe_df[col] = nd.astype(self.dtype)

        maybe_df.pipeline = maybe_pipeline.add(self).append_encoded_cols(self.params.keys())
        maybe_df.target = maybe_target
        if not self.inplace:
            return maybe_df

    def _save_uniques(self, col, uniques):
        """
        Saves provided unique values for a given column

        :param col: column for which the uniques should be
        saved
        :param uniques: unique values for the column
        """
        if uniques.dtype == object:
            missing_values_set = {
                value for value in uniques if value is None or is_scalar_nan(value)
            }
            if not missing_values_set:
                self.params[col] = np.sort(uniques)
            else:
                uniques = sorted(set(uniques) - missing_values_set)
                uniques.extend(missing_values_set)
                self.params[col] = np.array(uniques, dtype=object)
        else:
            self.params[col] = np.sort(uniques)

    def equals(self, other):
        return _BaseEncoder.equals(self, other) \
               and (self.unknown_value == other.unknown_value
                    or (is_scalar_nan(self.unknown_value) and is_scalar_nan(other.unknown_value))) \
               and (self.encoded_missing_value == other.encoded_missing_value
                    or (is_scalar_nan(self.encoded_missing_value) and is_scalar_nan(other.encoded_missing_value)))


class OneHotEncoder(_BaseEncoder):
    """
    Encodes categorical columns as onehot values.

    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be encoded too
    :param auto: if True, the transform will only be attempted for columns that can
    be identified as categorical. Columns are identified as categorical if there are
    20 or fewer values and the share of distinct values is less than 10%
    :param inplace: if True, the transform will be completed inplace
    :param dtype: the type to which the columns will be encoded
    :param categories: the values of the categories in each column to be
    encoded as a dictionary
    :param handle_unknown: the strategy to deal with not previously en-
    countered values in the encoder. Can be 'error' or 'use_encoded_value'. If
    'error', throws an error if an unknown category is encountered. If
    'use_encoded_value', replaces the new with unknown_value
    :param drop: can be None, 'first' or 'if_binary'. If 'first', drops the
    first encoded column. If 'if_binary', drops the first column only if the
    feature is binary. If None, doesn't drop any columns
    :param min_frequency: frequency of distinct value to not be considered an
    infrequent column, can be between 0 and 1 or an integer
    :param max_categories: if there are more than max_categories distinct values
    in a feature, n - max_categories of the least frequent values will be
    considered infrequent
    :param sparse: if True, the data will be encoded to sparse columns
    :param infix: the separator to put into the encoded column names
    between the original column name and the value
    :param keep_original: if True, the original columns will not be dropped
    """
    ALLOWED_HANDLE_UNKNOWN = {'error', 'ignore', 'infrequent_if_exist'}
    REQUIRES_SORTING = False
    INFREQUENT_CODE = 'infrequent'

    def __init__(self,
                 include: set | list | None = None,
                 exclude: set | list | None = None,
                 include_target: bool = False,
                 auto: bool = True,
                 inplace: bool = False,
                 dtype: Dtype = np.float64,
                 categories: str | dict = 'auto',
                 handle_unknown: str = 'error',
                 drop: str = None,
                 min_frequency: numbers.Number | None = None,
                 max_categories: numbers.Number | None = None,
                 sparse: bool = False,
                 infix: str = '_',
                 keep_original: bool = False):
        _BaseEncoder.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                              include_target=include_target, categories=categories, dtype=dtype,
                              handle_unknown=handle_unknown)
        self.drop = drop

        if not ((min_frequency is None) or (isinstance(min_frequency, numbers.Integral) and min_frequency >= 1) or (
                isinstance(min_frequency, numbers.Real) and (0.0 < min_frequency < 1.0))):
            raise ValueError("min_frequency must be an integer at least "
                             "1 or a float in range (0.0, 1.0); got the "
                             f"integer {min_frequency}")
        self.min_frequency = min_frequency

        if max_categories is not None and max_categories < 1:
            raise ValueError('max_categories cannot be less than 1')
        self.max_categories = max_categories

        self.sparse = sparse
        if is_integer_dtype(dtype):
            self.fill_value = 0
        elif dtype == np.dtype(bool):
            self.fill_value = False
        else:
            self.fill_value = 0.0

        self.infix = infix
        self.keep_original = keep_original

        self._check_infrequent = not (min_frequency is None and max_categories is None)
        self.params = None
        self.infrequents = None
        self._infrequent_representative = None  # ix lookup for numeric columns
        self.columns = {}

    def transform(self, df, ensure_validity: bool = True):
        """
        Transforms the provided dataframe.

        :param df: dataframe to be transformed
        :param ensure_validity: if True, will attempt to find values
        that were not encountered during training. Should be set to
        False only if the encoder was fitted to the same data it is
        transforming
        :return: transformed dataframe
        """
        self.validate_fit()
        maybe_df = df if self.inplace else df.copy(deep=True)
        maybe_pipeline = maybe_df.pipeline
        maybe_target = maybe_df.target
        with Lock.lock():
            for col in self.params:
                nd = maybe_df[col].to_numpy(copy=self.keep_original)
                uniques = unique(nd)
                handle_infrequent = self.infrequents and col in self.infrequents
                if nd.dtype.kind in "OUS":  # object columns are handled via python dictionary
                    diff = object_diff(uniques, self.params[col]) if ensure_validity else np.empty([0], dtype=nd.dtype)
                    if diff and self.handle_unknown == 'error':
                        raise RuntimeError(f'OrdinalEncoder.transform failed - unseen category '
                                           f'in the column {str(col)}')

                    if handle_infrequent:
                        dct, max_ix = self._handle_infrequent_object(col)
                        if self.handle_unknown == 'ignore':
                            for key in diff:
                                dct[key] = max_ix
                        else:
                            for key in diff:
                                dct[key] = -2

                    else:
                        dct, max_ix = self._handle_object(col)
                        if diff:
                            for key in diff:
                                dct[key] = max_ix

                    table = _nandict(dct)
                    nd = np.array([table[v] for v in nd])

                    self._add_columns_from_object_ix(maybe_df, col, nd, max_ix)

                else:  # numeric columns are handled via numpy
                    diff = np.setdiff1d(uniques, self.params[col], assume_unique=True) \
                        if ensure_validity else np.empty([0], dtype=nd.dtype)
                    diff = OneHotEncoder._reduce_nan(diff, self.params[col],
                                                     self.infrequents[col] if handle_infrequent else None)
                    handled_unknown = False
                    mask_unknown = None

                    if handle_infrequent:
                        mask_infrequent = np.isin(nd, self.infrequents[col])
                        if diff.size and self.handle_unknown == 'infrequent_if_exist':
                            mask_infrequent = mask_infrequent | np.isin(nd, diff) | np.isnan(nd) if np.isnan(
                                diff).any() else mask_infrequent | np.isin(nd, diff)
                            handled_unknown = True
                        if np.isnan(self.infrequents[col]).any():
                            mask_infrequent = mask_infrequent | np.isnan(nd)

                        nd[mask_infrequent] = self._infrequent_representative[col][1]

                    if diff.size:
                        if self.handle_unknown == 'error':
                            raise RuntimeError('OneHotEncoder.transform failed - '
                                               f'unseen category in the column {str(col)}')

                        elif not handled_unknown:  # we land here if we need to ignore unknown
                            mask_unknown = np.isin(nd, diff)
                            nd[mask_unknown] = self.params[col].shape[0]  # will be the last column

                    nd = np.searchsorted(self.params[col], nd)
                    if mask_unknown is not None:
                        nd[mask_unknown] = self.params[col].shape[0]
                    self._add_columns_from_numeric_ix(maybe_df, col, nd, self.params[col].shape[0], handle_infrequent)

                maybe_pipeline.append_encoded_cols(self.columns[col])

            if not self.keep_original:
                with maybe_df.pipeline:
                    maybe_df.drop(labels=self.params.keys(), axis=1, inplace=True)

        maybe_df.pipeline = maybe_pipeline.add(self)
        maybe_df.target = maybe_target
        if not self.inplace:
            return maybe_df

    def _fit(self, df, columns: list):
        """
        Fits itself to the provided dataframe. Calculates
        the unique and, if required, infrequent values for each
        column

        :param df: dataframe to be fitted
        :param columns: columns to be fitted
        """
        self.columns = {}  # overwrite anything that was here before
        if self._check_infrequent:
            infrequent_border = self._determine_infrequent_border(df.shape[0])
            self.params = {}
            self.infrequents = {}
            self._infrequent_representative = {}
            for col in columns:
                uniques, counts = self._determine_unique(df[col].to_numpy(), return_counts=True)
                self._check_infrequents_and_save(col, uniques, counts, infrequent_border)
                self._save_uniques(col, unique(df[col]))
        else:
            _BaseEncoder._fit(self, df, columns)

    def _autofit(self, df, columns: list):
        """
        Fits itself to the provided dataframe. Calculates
        the unique and, if required, infrequent values for each
        column, if the column can be determined as categorical

        :param df: dataframe to be fitted
        :param columns: columns to be fitted
        """
        self.columns = {}  # overwrite anything that was here before
        if self._check_infrequent:
            border = min(OrdinalEncoder.CATEGORICAL_NUMBER, OneHotEncoder.CATEGORICAL_SHARE * df.shape[0])
            infrequent_border = self._determine_infrequent_border(df.shape[0])
            self.params = {}
            self.infrequents = {}
            self._infrequent_representative = {}
            for col in columns:
                uniques, counts = self._determine_unique(df[col].to_numpy(), return_counts=True)
                if len(uniques) <= border or (self.categories != 'auto' and col in self.categories):
                    self._check_infrequents_and_save(col, uniques, counts, infrequent_border)
                    self._save_uniques(col, unique(df[col]))
        else:
            _BaseEncoder._autofit(self, df, columns)

    def _save_uniques(self, col, uniques: np.ndarray):
        """
        Saves unique feature values for a column. Handles
        "OUS" and not-"OUS" columns differently. For "OUS"
        columns, most of the infrequent handling will be
        completed during transform

        :param col: column for which the data is saved
        :param uniques: the unique values
        """

        if uniques.dtype.kind in "OUS":
            self.params[col] = uniques
        else:
            uniques = np.sort(uniques)
            if self.infrequents and col in self.infrequents:
                mask_infrequent = np.isin(uniques, self.infrequents[col]) | np.isnan(uniques) if np.isnan(
                    self.infrequents[col]).any() else np.isin(uniques, self.infrequents[col])
                infrequent_ix = np.nonzero(mask_infrequent)[0][0]
                mask_infrequent[infrequent_ix] = False  # don't want to delete this one
                self._infrequent_representative[col] = (infrequent_ix, uniques[infrequent_ix])
                self.params[col] = uniques[~mask_infrequent]
                self.columns[col] = [
                    str(col) + self.infix + str(
                        x) if i != infrequent_ix else str(col) + self.infix + OneHotEncoder.INFREQUENT_CODE for
                    i, x in enumerate(self.params[col])]
            else:
                self.params[col] = uniques
                self.columns[col] = [str(col) + self.infix + str(x) for x in self.params[col]]

    def _check_infrequents_and_save(self, col, uniques: np.ndarray, counts: np.ndarray, infrequent_border: int):
        """
        Saves values that will be considered infrequent for non-OUS columns

        :param col: column for which the infrequent values are saved
        :param uniques: unique values for the column
        :param counts: counts of unique values for the column
        :param infrequent_border: if the value occurs in the data less than
        infrequent_border times, it will be considered infrequent
        """
        if infrequent_border != -1 or (self.max_categories is not None and self.max_categories < uniques.shape[0]):
            argsorted = counts.argsort()
            counts_sorted = counts[argsorted]
            uniques_sorted = uniques[argsorted]
            infrequent_border_ix = 0 if infrequent_border == -1 else np.searchsorted(counts_sorted,
                                                                                     infrequent_border,
                                                                                     side='left')
            infrequent_cats = max(uniques.shape[0] - self.max_categories, 0) if self.max_categories is not None else 0
            final_border = max(infrequent_cats, infrequent_border_ix)
            if final_border > 0:
                infrequents = uniques_sorted[:final_border]
                self.infrequents[col] = infrequents

    def _determine_infrequent_border(self, n_rows: int):
        """
        Determines the minimal number of times the value
        must appear in the data to not be considered infrequent

        :param n_rows: the number of rows in the dataframe
        :return: the border, if -1, all values should be considered
        frequent if max_category is not specified
        """
        if self.min_frequency is not None:
            if isinstance(self.min_frequency, numbers.Integral):
                return self.min_frequency
            else:
                return self.min_frequency * n_rows
        return -1

    def _handle_infrequent_object(self, col):
        """
        Adds the OUS data to a dict to be mapped to separate columns
        later on. Handles infrequent values.

        :param col: column to be transformed
        :return: dict of value : column pairs, number of columns
        to be encoded
        """
        dct = {}
        correction = 0
        i = -1
        columns = []
        infrequents_nan = pd.isnull(self.infrequents[col]).any()
        for i, val in enumerate(self.params[col]):
            if val in self.infrequents[col] or (is_scalar_nan(val) and infrequents_nan):
                dct[val] = -2
                correction += 1
            else:
                dct[val] = i - correction
                columns.append(str(col) + self.infix + str(val))

        columns.append(str(col) + self.infix + OneHotEncoder.INFREQUENT_CODE)
        self.columns[col] = columns
        return dct, i - correction + 2  # +1 column for infrequents, +1 column for ignored values

    def _handle_object(self, col):
        """
        Adds the OUS data to a dict to be mapped to separate columns
        later on

        :param col: column to be transformed
        :return: dict of value : column pairs, number of columns
        to be encoded
        """
        dct = {}
        i = -1
        for i, val in enumerate(self.params[col]):
            dct[val] = i
        self.columns[col] = [str(col) + self.infix + str(item) for item in dct]
        return dct, i + 1  # +1 column for ignored values

    def _add_columns_from_object_ix(self, df, col, nd: np.ndarray, max_ix: int):
        """
        Adds onehot-encoded columns for a given OUS column

        :param df: dataframe to be transformed
        :param col: column that is being encoded
        :param nd: numpy array with values to be encoded
        :param max_ix: the number of onehot columns to be
        created
        """
        if self.sparse:
            for i in range(max_ix):
                if not self._drop_first(max_ix) or i != 0:
                    if i < max_ix - 1:
                        mask = nd == i
                    else:
                        mask = (nd == i) | (nd == -2)  # we can have -2 as a valid value
                    df[self.columns[col][i]] = self._create_sparse_from_mask(mask)
        else:
            nd = np.eye(max_ix + 1, dtype=self.dtype).take(nd, axis=1).T[:, :-1]
            for i, column in enumerate(self.columns[col]):
                if not self._drop_first(max_ix) or i != 0:
                    df[column] = nd[:, i]

    def _add_columns_from_numeric_ix(self, df, col, nd: np.ndarray, max_ix: int, handle_infrequent: bool):
        """
        Adds onehot-encoded columns for a given non-OUS column

        :param df: dataframe to be transformed
        :param col: column that is being encoded
        :param nd: numpy array with values to be encoded
        :param max_ix: the number of onehot columns to be
        created
        :param handle_infrequent: if the infrequent values should be handled
        """
        # here, the index of the infrequent column is actually saved in the self.infrequent_representatives[col][0]
        drop_first = self._drop_first(max_ix)

        if self.sparse:
            if handle_infrequent:
                infrequent_ix = self._infrequent_representative[col][0]
                for i in range(max_ix):
                    if (i != 0 or not drop_first) and i != infrequent_ix:
                        df[self.columns[col][i]] = self._create_sparse_from_mask(nd == i)
                if infrequent_ix != 0 or not drop_first:
                    df[self.columns[col][infrequent_ix]] = self._create_sparse_from_mask(nd == infrequent_ix)

            else:
                for i in range(max_ix):
                    if i != 0 or not drop_first:
                        df[self.columns[col][i]] = self._create_sparse_from_mask(nd == i)

        else:
            nd = np.eye(max_ix + 1, dtype=self.dtype).take(nd, axis=1).T[:, :-1]
            if handle_infrequent:
                infrequent_ix = self._infrequent_representative[col][0]

                for i, column in enumerate(self.columns[col]):
                    if i != infrequent_ix and (i != 0 or not drop_first):
                        df[column] = nd[:, i]
                if infrequent_ix != 0 or not drop_first:
                    df[self.columns[col][infrequent_ix]] = nd[:, infrequent_ix]

            else:
                for i, column in enumerate(self.columns[col]):
                    if i != 0 or not drop_first:
                        df[column] = nd[:, i]

    def _create_sparse_from_mask(self, mask: np.ndarray):
        """
        Creates a pd.SparseArray from a numpy mask

        :param mask: ndarray of values to be encoded
        :return:
        """
        return pd.arrays.SparseArray(mask, dtype=self.dtype)

    def _drop_first(self, max_ix: int):
        """
        returns True if the first column should be dropped,
        False otherwise

        :param max_ix: number of columns that will be encoded
        :return: True / False, see above
        """
        return self.drop is not None and (self.drop == 'first' or (self.drop == 'if_binary' and max_ix == 2))

    def equals(self, other):
        return _BaseEncoder.equals(self, other) \
               and self.drop == other.drop \
               and self.handle_unknown == other.handle_unknown \
               and self.min_frequency == other.min_frequency \
               and self.max_categories == other.max_categories \
               and self.keep_original == other.keep_original \
               and self.infix == other.infix \
               and self.sparse == other.sparse

    @staticmethod
    def _determine_unique(nd: np.ndarray, return_counts: bool):
        """
        determines unique values if the counts are needed

        :param nd: feature array
        :param return_counts: if the counts should be returned
        :return: unique values
        """
        if nd.dtype.kind in 'OUS':
            return _unique_python(nd, return_counts=return_counts)
        else:
            return np.unique(nd, return_counts=return_counts)
