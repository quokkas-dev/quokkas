from __future__ import annotations

from abc import abstractmethod

import numpy as np
import pandas as pd

from .generic import _BaseExplainer
from ..utils.graphic_utils import Plotter


class ContinuousFeatureImportanceEstimator(_BaseExplainer):
    """
    Calculates feature importance as variance of feature means
    among buckets of values less expected variance for independent
    feature values. Buckets are formed as feature values for n quantiles
    of target values.

    :param buckets: the number of target quantiles to be considered
    :param include: if provided, only those features will be analysed
    :param exclude: if provided, these features won't be analysed
    :param auto: if True, only numeric columns will be considered
    """

    def __init__(self, buckets=100, include=None, exclude=None, auto=True):
        _BaseExplainer.__init__(self, include=include, exclude=exclude, auto=auto, include_target=True)
        self.buckets = buckets
        self.params = None
        self.params_dict = None
        self.target_name = None

    def fit(self, df, target=None):
        """
        Calculates the feature importance

        :param df: dataframe to be fitted
        :param target: if provided, this column will be used as target.
        Otherwise, the target of the dataframe will be used
        """
        self.target_name = self.determine_unique_target(df, target)

        cpy = df.sort_values(by=self.target_name, axis=0, inplace=False)

        target = cpy.pop(self.target_name).to_numpy()

        trgts = np.array_split(target, self.buckets)
        target_means = np.stack([np.nanmean(trgt, axis=0) for trgt in trgts])

        cols = self._select_numeric_columns(cpy)
        cpy = cpy.loc[:, cols]

        nd = cpy.to_numpy()
        del cpy
        nd = (nd - np.nanmean(nd, axis=0)) / (np.nanstd(nd, axis=0) + 1e-12)

        corr = 1 / np.sqrt(nd.shape[0] // self.buckets)  # expected std of means

        nds = np.array_split(nd, self.buckets)
        del nd
        statistics = np.stack([np.nanmean(nd, axis=0) for nd in nds])
        del nds

        self.params = np.maximum(np.nanstd(statistics, axis=0) - corr, 0)
        sum_ = self.params.sum()
        self.params = self.params / sum_ if sum_ > np.finfo(statistics.dtype).eps*10 else np.ones_like(self.params) / self.params.shape[0]
        self.params_dict = {column: value for column, value in zip(cols, self.params)}

        statistics = pd.DataFrame(statistics, columns=cols)
        statistics[self.target_name] = target_means
        self.statistics = pd.melt(statistics, [self.target_name], var_name='Params', value_name='Standardized Params')

        self.fitted = True

    def visualize(self, figsize=(12, 5), style='seaborn'):
        """
        Plots means of the features split by quantiles of the target
        variable

        :param figsize: the size of the figure that it will be plotted on
        :param style: one of the available plt styles
        """
        sns, ax = Plotter.initialize(style=style, figsize=figsize, title='Feature Means')

        sns.lineplot(data=self.statistics, x=self.target_name, y='Standardized Params', hue='Params',
                     style='Params', ax=ax)
        Plotter.plot(legend=True)

    def __repr__(self):
        return "FeatureImportanceEstimator: " + repr(self.params_dict)

    def to_dict(self):
        return self.params_dict

    def to_numpy(self):
        return self.params


class CategoricalFeatureImportanceEstimator(_BaseExplainer):
    """
    Calculates feature importance as variance of feature means
    among buckets of values less expected variance for independent
    feature values. Buckets are formed for each distinct target
    value

    :param include: if provided, only those features will be analysed
    :param exclude: if provided, these features won't be analysed
    :param auto: if True, only numeric columns will be considered
    """

    def __init__(self, include=None, exclude=None, auto=True):
        _BaseExplainer.__init__(self, include=include, exclude=exclude, auto=auto, include_target=True)
        self.params = None
        self.params_dict = None

    def fit(self, df, target=None):
        """
        Calculates the feature importance

        :param df: dataframe to be fitted
        :param target: if provided, this column will be used as target.
        Otherwise, the target of the dataframe will be used
        """
        self.target_name = self.determine_unique_target(df, target)
        unique_targets = pd.unique(df[self.target_name])
        ixs = [(df[self.target_name] == trgt).to_numpy() for trgt in unique_targets]
        cpy = df.drop(self.target_name, axis=1, inplace=False)
        cols = self._select_numeric_columns(cpy)
        cpy = cpy.loc[:, cols]

        nd = cpy.to_numpy()
        del cpy

        nd = (nd - np.nanmean(nd, axis=0)) / (np.nanstd(nd, axis=0) + 1e-12)
        nds = [nd[ix] for ix in ixs]

        corr = 1 / np.sqrt(nd.shape[0] // len(unique_targets))
        statistics = np.stack([np.nanmean(nd, axis=0) for nd in nds])
        del nd, nds

        self.params = np.maximum(np.nanstd(statistics, axis=0) - corr, 0)
        sum_ = self.params.sum()
        self.params = self.params / sum_ if sum_ > np.finfo(statistics.dtype).eps*10 else np.ones_like(self.params) / self.params.shape[0]
        self.params_dict = {column: value for column, value in zip(cols, self.params)}

        statistics = pd.DataFrame(statistics, columns=cols)
        statistics[self.target_name] = unique_targets
        self.statistics = pd.melt(statistics, [self.target_name], value_name='Standardized Params', var_name='Params')

        self.fitted = True

    def visualize(self, figsize=(10, 5), style='seaborn'):
        """
        Plots means of the features split by distinct values of the
        target variable

        :param figsize: the size of the figure that it will be plotted on
        :param style: one of the available plt styles
        """
        sns, ax = Plotter.initialize(style=style, figsize=figsize, title='Feature Means')
        sns.barplot(data=self.statistics, x=self.target_name, y='Standardized Params', hue='Params',
                    ax=ax)
        Plotter.plot(legend=True)

    def __repr__(self):
        return "FeatureImportanceEstimator: " + repr(self.params_dict)

    def to_dict(self):
        return self.params_dict

    def to_list(self):
        return list(self.params)

    def to_numpy(self):
        return self.params
