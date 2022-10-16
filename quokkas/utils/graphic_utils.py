from __future__ import annotations
import importlib


class Plotter:
    """
    A helper class for figure plotting.
    It imports the required dependencies and provides
    a simple interface for plotting of the charts.
    """
    sns = None
    plt = None

    @classmethod
    def initialize(cls, style: str = 'seaborn', figsize: tuple = (12, 5), subplot_shape: tuple = (1, 1), title: str | None = None):
        """
        Initializes the plotting chart. Imports the dependencies
        (if not already imported), sets style and creates a subplots
        figure. Returns sns (as an imported module) and the reference
        to the figure axes (to be filled by caller)

        :param style: one of available plt styles
        :param figsize: subplot figure size
        :param subplot_shape: shape of subplots
        :param title: title to be set
        :return: tuple with imported seaborn and axes of the subplot
        """

        if cls.plt is None:
            cls.plt = importlib.import_module('matplotlib.pyplot')
        if cls.sns is None:
            cls.sns = importlib.import_module('seaborn')

        cls.plt.style.use(style)
        _, ax = cls.plt.subplots(*subplot_shape, figsize=figsize)

        if title is not None:
            cls.plt.suptitle(title, y=0.99)

        return cls.sns, ax

    @classmethod
    def plot(cls, ax=None, ylabel=None, xlabel=None, legend=False):
        """
        Plots the provided graph

        :param ax: axes of the subplot figure
        :param ylabel: ylabel for the axes
        :param xlabel: xlabel for the axes
        :param legend: if the legend should be provided
        """


        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if legend:
            cls.plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

        cls.plt.tight_layout()
        cls.plt.show()



