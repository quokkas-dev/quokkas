from __future__ import annotations

import numpy as np

from ..core.pipeline.lock import Lock
from .generic import _BaseProcessor


class DateEncoder(_BaseProcessor):
    """
    Encodes the date columns in the dataset. Allows to map the dates
    to a [0, 1] interval, map the days in a year into a [0, 1] interval
    (to account for seasonal effects) and map the day of the week to a
    [0, 6] interval (to account for weekday effects).

    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be encoded too
    :param auto: if True, the transform will only be attempted for datetime columns
    :param inplace: if True, the transform will be completed inplace
    :param ordinal: if the dates should be encoded ordinally, i.e. maximal
    date mapped to one, minimal to 0, all others linearly between them
    :param intrayear: if the days of year should be encoded, i.e.
    31.12. 23:59:59 to 1, 01.01. 00:00:00 to 0, all others linearly between them
    :param intraweek: if the days should be added as day of the week (0 - 6)
    :param keep_original: if the original date columns should be saved
    """

    YEAR_LENGTH = 31536000000000000
    DAY_LENGTH = 86400000000000
    WEEK_LENGTH = 604800000000000

    def __init__(self, inplace: bool = False, auto: bool = True, include: set | list | None = None,
                 exclude: set | list | None = None, include_target: bool = False, ordinal: bool = True,
                 intrayear: bool = False, intraweek: bool = False, keep_original: bool = False):

        _BaseProcessor.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                                include_target=include_target)

        self.ordinal = ordinal
        self.intrayear = intrayear
        self.intraweek = intraweek
        self.keep_original = keep_original

        self.params = None
        self.cols = None

    def fit(self, df):
        """
        Fits the date encoder to the provided data. In particular,
        saves the columns that are to be transformed, and gets
        max / min for each column if ordinal encoding is requested

        :param df: data to be fitted
        """
        with Lock.lock():

            self.cols = self._select_date_columns(df)
            if self.ordinal:
                self.params = []
                for col in self.cols:
                    # if pandas changes native representation from datetime64[ns] only, we'll have a problem
                    # usually faster to convert to numpy, but depends on the layout of data
                    ordinal = df[col].to_numpy().astype(np.int64)
                    amin = np.nanmin(ordinal)
                    amax = np.nanmax(ordinal)
                    self.params.append((amin, amax - amin if amin != amax else 1))

        self.fitted = True

    def _select_date_columns(self, df):
        """
        Selects columns to be transformed. If auto, additionally
        filters out not datetime columns.

        :param df: dataframe to be transformed
        """
        cols = self._select_columns(df)

        if self.auto:
            cols = df.pipeline.reduce_encoded_cols(cols)
            cols = cols.intersection(df.select_dtypes(include=['datetime64']).columns)

        return [col for col in df.columns if col in cols]  # preserve order

    def transform(self, df):
        """
        Encodes the datetime columns in 1-3 ordinal columns.

        :param df: dataframe to be transformed
        :return: transformed dataframe
        """

        maybe_df = df if self.inplace else df.copy(deep=True)
        maybe_pipeline = maybe_df.pipeline

        with Lock.lock():
            for i, col in enumerate(self.cols):
                nd = maybe_df[col].to_numpy()
                ordinal = nd.astype(np.int64)

                if self.ordinal:
                    col_name = str(col) + '_ordinal'
                    maybe_df[col_name] = (ordinal - self.params[i][0]) / (self.params[i][1])
                    maybe_pipeline.add_encoded_col(col_name)

                if self.intrayear:
                    is_leap = nd.astype('datetime64[Y]').astype(int) % 4 == 0
                    col_name = str(col) + '_intrayear'
                    maybe_df[col_name] = (nd - nd.astype('datetime64[Y]')).astype(np.int64) / (
                                self.YEAR_LENGTH + self.DAY_LENGTH * is_leap)
                    maybe_pipeline.add_encoded_col(col_name)

                if self.intraweek:
                    col_name = str(col) + '_intraweek'
                    maybe_df[col_name] = maybe_df[col].dt.dayofweek
                    maybe_pipeline.add_encoded_col(col_name)

            if not self.keep_original:
                maybe_df.drop(self.cols, axis=1, inplace=True)

        maybe_df.pipeline = maybe_pipeline.add(self)

        if not self.inplace:
            return maybe_df

    def equals(self, other):
        return _BaseProcessor.equals(self, other) \
               and self.ordinal == other.ordinal \
               and self.keep_original == other.keep_original \
               and self.intrayear == other.intrayear \
               and self.intraweek == other.intraweek
