from __future__ import annotations
from abc import abstractmethod

from ..transforms.generic import _BaseColumnSelector


class _BaseExplainer(_BaseColumnSelector):
    """
    Base class for the EDA functions. Provides basic column selection
    functionality (via _BaseColumnSelector).

    :param include: if provided, only those features will be considered
    :param exclude: if provided, these features won't be considered
    :param auto: if True, only numeric columns will be considered
    :param include_target: if True, the target will be considered as well
    """

    def __init__(self, include=None, exclude=None, auto=True, include_target=False):
        _BaseColumnSelector.__init__(self, include=include, exclude=exclude, auto=auto, include_target=include_target)
        self.fitted = False

    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def visualize(self):
        pass

    def _select_numeric_columns(self, df) -> list:
        """
        Selects numeric columns in the original dataframe columns
        order

        :param df: dataframe to be fitted
        :return: list of columns
        """
        return [col for col in df.columns if col in _BaseColumnSelector._select_numeric_columns(self, df)]

    @classmethod
    def determine_unique_target(cls, df, target):
        """
        Determines unique target based on the provided dataframe and
        an additional argument target (which might be None)

        :param df: dataframe to be fitted
        :param target: if provided, this column will be used as target.
        Otherwise, the target of the dataframe will be used
        """

        if target is None and not df.target:
            raise ValueError('Target not specified. '
                             'Please provide a target to the fit '
                             'function or specify dataframe target '
                             'via DataFrame.targetize')

        if target is not None and df.target and target not in df.target:
            raise ValueError('Provided target does not agree '
                             'with the dataframe target')

        if target is None:
            if len(df.target) > 1:
                raise ValueError(f'Currently, {cls.__name__}'
                                 ' supports only singular targets')
            return next(iter(df.target))

        if df.target and target not in df.target:
            raise ValueError('Provided target does not agree '
                             'with the dataframe target')
        return target


