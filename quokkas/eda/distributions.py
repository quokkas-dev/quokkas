import warnings

import numpy as np

from .categorical import CategoricalDetector
from .generic import _BaseExplainer
from ..utils.graphic_utils import Plotter
from ..utils.pd_utils.pd_utils import find_stack_level


class DistVisualizer(_BaseExplainer):
    """
    Plot estimated density distribution of the non-categorical features
    for distinct target values. Should be only applied for categorical
    targets.

    :param include: if provided, only those features will be plotted
    :param exclude: if provided, these features won't be plotted
    :param auto: if True, only numeric columns will be considered
    :param cat_detection_strategy: 'count', 'type' or 'count&type'. If
    'count', the categorical features will be detected based on the
    number of distinct values (with border being min(cat_share * num_rows,
    cat_number)), if 'type' - based on the type of the column, if
    'count&type' - based on both
    :param cat_number: if 'count' or 'count&type', the features with less
    than this number of distinct values will be considered categorical (if
    cat_share * num_rows is larger than cat_number)
    :param cat_share: if 'count' or 'count&type', the features with less
    than cat_share * num_rows of distinct values will be considered categorical
    (if cat_number is larger than cat_share * num_rows)
    """
    RECOMMENDED_MAX_SIZE = 10  # raise warning if larger
    TOTAL_MAX_SIZE = 100  # throw an error if larger
    DEFAULT_WIDTH = 12

    def __init__(self, include=None, exclude=None, auto=True, cat_detection_strategy='count',
                 cat_number=CategoricalDetector.DEFAULT_CAT_NUMBER, cat_share=CategoricalDetector.DEFAULT_CAT_SHARE):
        _BaseExplainer.__init__(self, include=include, exclude=exclude, auto=auto, include_target=False)
        CategoricalDetector.validate_strategy(cat_detection_strategy)
        self.cat_detection_strategy = CategoricalDetector.ALLOWED_STRATEGIES.index(cat_detection_strategy)
        self.cat_number = cat_number
        self.cat_share = cat_share

        self.df = None
        self.columns = None
        self.target_name = None

    def fit(self, df, target=None):
        """
        Determines the columns for which the density estimations will
        be plotted

        :param df: dataframe to be fitted
        :param target: if not None, the feature densities will be estimated
        for the unique values of this column
        """
        self.target_name = self.determine_unique_target(df, target)
        self.df = df  # just a ref to the original df

        n_unique = df[self.target_name].nunique(dropna=False)

        if n_unique > self.TOTAL_MAX_SIZE:
            raise ValueError(f"The target column contains more values than allowed ({self.TOTAL_MAX_SIZE})")

        if df[self.target_name].nunique(dropna=False) > self.RECOMMENDED_MAX_SIZE:
            warnings.warn(
                "The target column contains more values than the recommended max size"
                f"of {self.RECOMMENDED_MAX_SIZE}",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        columns = self._select_columns(df)
        if self.auto:
            columns -= set(CategoricalDetector.determine_categorical(df, columns, self.cat_detection_strategy,
                                                                     cat_number=self.cat_number,
                                                                     cat_share=self.cat_share))
        self.columns = [col for col in df.columns if col in columns]
        self.fitted = True

    def visualize(self, figsize=None, style='seaborn', n_line_items=3, legend=True, **kwargs):
        """
        Plots density estimations for each distinct target value. Internally,
        it utilizes the sns.kdeplot functionality

        :param figsize: the size of the figure that it will be plotted on
        :param style: one of the available plt styles
        :param n_line_items: number of plots in one line
        :param legend: if the legend should be included
        :param kwargs: additional kw arguments to be passed to the plot
        """
        quot, rem = divmod(len(self.columns), n_line_items)
        subplot_shape = ((quot + 1 if rem else quot), (n_line_items if quot else rem))
        figsize = (self.DEFAULT_WIDTH, self.DEFAULT_WIDTH / n_line_items * quot) if figsize is None else figsize

        sns, axes = Plotter.initialize(style=style, figsize=figsize, subplot_shape=subplot_shape,
                                       title='Distributions of Features')
        target = self.df[self.target_name]

        for ax, col in zip(axes.flatten(order='C') if isinstance(axes, np.ndarray) else [axes], self.columns):
            sns.kdeplot(x=self.df[col], hue=target, ax=ax, legend=legend, **kwargs)
            ax.set_ylabel(None)

        Plotter.plot()
