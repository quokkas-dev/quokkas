from .generic import _BaseExplainer
from ..utils.graphic_utils import Plotter


class MissingValuesVisualizer(_BaseExplainer):
    """
    Plots missing / non-missing values share per each feature.

    :param include: if provided, only those features will be plotted
    :param exclude: if provided, these features won't be plotted
    :param auto: if True, only numeric columns will be considered
    """
    def __init__(self, include=None, exclude=None, auto=False):
        _BaseExplainer.__init__(self, include=include, exclude=exclude, auto=auto, include_target=True)
        self.missing = None

    def fit(self, df):
        """
        Calculates the number of missing values for each feature

        :param df: dataframe to be fitted
        """
        cols = self._select_numeric_columns(df)
        df = df.loc[:, cols]
        self.missing = df.isna().sum(axis=0) / df.shape[0]
        self.missing.sort_values(inplace=True, ascending=False)

        self.fitted = True

    def visualize(self, figsize=(12, 5), reverse=False, style='seaborn', **kwargs):
        """
        Plots missing values share per each feature in a barplot. If
        reverse, plots the share of non-missing values instead

        :param reverse: if True, the share non-missing values will be plotted
        :param figsize: the size of the figure that it will be plotted on
        :param style: one of the available plt styles
        :param kwargs: additional kw arguments to be provided to the chart
        :return:
        """
        if reverse:
            sns, ax = Plotter.initialize(style=style, figsize=figsize, title='Data Availability')
            sns.barplot(x=self.missing.index, y=1 - self.missing.values, ax=ax, **kwargs)
            ax.tick_params(axis='x', rotation=90)
            Plotter.plot(ax=ax, ylabel='Share of available data', xlabel='Features')
        else:
            sns, ax = Plotter.initialize(style=style, figsize=figsize, title='Missing Data')
            sns.barplot(x=self.missing.index, y=self.missing.values, ax=ax, **kwargs)
            ax.tick_params(axis='x', rotation=90)
            Plotter.plot(ax=ax, ylabel='Share of missing data', xlabel='Features')

