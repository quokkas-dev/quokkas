import numpy as np
from .generic import _BaseExplainer
from ..utils.graphic_utils import Plotter


class CorrelationVisualizer(_BaseExplainer):
    """
    Plots correlation coefficients of the features.

    :param include: if provided, only those features will be plotted
    :param exclude: if provided, these features won't be plotted
    :param auto: if True, only numeric columns will be considered
    :param include_target: if True, the correlation with target will
    be plotted as well
    """

    def __init__(self, include=None, exclude=None, auto=True, include_target=True):
        _BaseExplainer.__init__(self, include=include, exclude=exclude, auto=auto, include_target=include_target)
        self.corrs = None

    def fit(self, df):
        """
        Calculates the correlation matrix of (if auto - numeric) features

        :param df: dataframe to be fitted
        """
        cols = self._select_numeric_columns(df)
        df = df.loc[:, cols]
        self.corrs = df.corr()
        self.fitted = True

    def visualize(self, figsize=(10, 10), style='seaborn', absolute=False, annot=False, cbar=True, cmap='coolwarm',
                  **kwargs):
        """
        Plots correlation coefficients on a heatmap. Can plot absolute /
        not absolute correlation coefficients depending on absolute parameter

        :param figsize: the size of the figure that it will be plotted on
        :param style: one of the available plt styles
        :param annot: if True, the annotations will be provided
        :param cbar: if True, the cbar will be shown on the heatmap
        :param cmap: cmap to be passed to the heatmap
        :param kwargs: additional arguments to be passed to the heatmap
        """
        sns, ax = Plotter.initialize(style=style, figsize=figsize, title='Feature Correlations')

        mask = kwargs.pop('mask') if 'mask' in kwargs else np.triu(np.ones(self.corrs.shape))
        vmin = kwargs.pop('vmin') if 'vmin' in kwargs else (0 if absolute else -1)
        vmax = kwargs.pop('vmax') if 'vmax' in kwargs else 1
        corrs = self.corrs.abs() if absolute else self.corrs

        sns.heatmap(corrs, ax=ax, mask=mask,
                    vmin=vmin, vmax=vmax, annot=annot, cbar=cbar, cmap=cmap, **kwargs)

        Plotter.plot(ax)

    def __repr__(self):
        return repr(self.corrs)

    def to_list(self):
        return self.corrs
