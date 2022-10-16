import numpy as np

from .generic import _BaseExplainer
from ..utils.graphic_utils import Plotter


class ScatterVisualizer(_BaseExplainer):
    """
    Creates scatter plots for each feature vs target

    :param include: if provided, only those features will be plotted
    :param exclude: if provided, these features won't be plotted
    :param auto: if True, only numeric columns will be considered
    """
    DEFAULT_WIDTH = 12

    def __init__(self, include=None, exclude=None, auto=True):
        _BaseExplainer.__init__(self, include=include, exclude=exclude, auto=auto, include_target=False)

        self.df = None
        self.columns = None
        self.target_name = None

    def fit(self, df, target=None):
        """
        Determines the feature columns which will be plotted

        :param df: dataframe to be fitted
        :param target: if provided, this column will be used as target.
        Otherwise, the target of the dataframe will be used
        """
        self.target_name = self.determine_unique_target(df, target)
        self.df = df  # just a ref to the original df

        self.columns = self._select_numeric_columns(df)

        self.fitted = True

    def visualize(self, figsize=None, style='seaborn', n_line_items=3, legend=False, **kwargs):
        """
        Creates scatterplots for each feature vs target

        :param n_line_items: number of plots in one line
        :param legend: if the legend should be included
        :param figsize: the size of the figure that it will be plotted on
        :param style: one of the available plt styles
        :param kwargs: additional kw arguments to be passed to the plot
        """
        quot, rem = divmod(len(self.columns), n_line_items)
        subplot_shape = ((quot + 1 if rem else quot), (n_line_items if quot else rem))
        figsize = (self.DEFAULT_WIDTH, self.DEFAULT_WIDTH / n_line_items * quot) if figsize is None else figsize

        sns, axes = Plotter.initialize(style=style, figsize=figsize, subplot_shape=subplot_shape,
                                       title='Feature Scatterplots')

        target = self.df[self.target_name]

        for ax, col in zip(axes.flatten(order='C') if isinstance(axes, np.ndarray) else [axes], self.columns):
            sns.scatterplot(x=self.df[col], y=target, ax=ax, legend=legend, **kwargs)
            ax.set_ylabel(None)

        Plotter.plot()
