from __future__ import annotations

import numpy as np

from ..core.pipeline.lock import Lock
from .generic import _BaseProcessor


class Normalizer(_BaseProcessor):
    """
    Normalizes the dataframe values row-wise. Supports l2, l1 and max (l_inf) norms.
    The normalization does not center data.

    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be normalized too
    :param auto: if True, the transform will only be attempted for numeric columns
    :param inplace: if True, the transform will be completed inplace
    :param norm: 'l2', 'l1' or 'max'
    """

    ALLOWED_NORMS = {'l2', 'l1', 'max'}

    def __init__(self, inplace: bool = False, auto: bool = True, include: set | list | None = None,
                 exclude: set | list | None = None, include_target: bool = False, norm: str = 'l2'):
        if norm not in self.ALLOWED_NORMS:
            raise ValueError(
                "received unknown norm parameter"
                f"{norm}. Expected one of:"
                f" {self.ALLOWED_NORMS}"
            )
        _BaseProcessor.__init__(self, inplace=inplace, auto=auto, include=include, exclude=exclude,
                                include_target=include_target)
        self.norm = norm
        self.columns = None

    def fit(self, df):
        """
        Fits the normalizer to the provided data. Since there are no
        parameters to be saved, we just keep track of the provided
        columns

        :param df: dataframe to be fitted
        """
        self.columns = self._select_numeric_columns(df)
        self.fitted = True

    def transform(self, df):
        """
        Normalizes the provided data in selected columns according
        to the saved norm.

        :param df: dataframe to be transformed
        :return: transformed dataframe
        """
        maybe_df = df if self.inplace else df.copy(deep=True)
        maybe_pipeline = maybe_df.pipeline
        maybe_target = maybe_df.target
        with Lock.lock():
            complete = self.columns == set(df.columns)
            if not complete:
                nd = maybe_df.loc[:, list(self.columns)].to_numpy()
            else:
                nd = maybe_df.to_numpy()

            if self.norm == 'l2':
                norms = np.nanstd(nd, axis=1)
            elif self.norm == 'l1':
                norms = np.nansum(np.abs(nd), axis=1)
            else:
                norms = np.nanmax(np.abs(nd), axis=1)

            nd /= norms[:, np.newaxis]

            if complete:
                maybe_df.reinitialize(nd)
            else:
                for i, col in enumerate(self.columns):
                    maybe_df[col] = nd[:, i]

        maybe_df.pipeline = maybe_pipeline.add(self)
        maybe_df.target = maybe_target

        if not self.inplace:
            return maybe_df

    def equals(self, other):
        return _BaseProcessor.equals(self, other) and \
               self.norm == other.norm



