from typing import Callable
from .generic import _BaseTransformer
from copy import deepcopy as _deepcopy


class Mapper(_BaseTransformer):
    """
    Executor class for the map function. Saves initially provided
    arguments, and, if no new arguments were provided to transform,
    re-utilizes them. Otherwise, utilizes the provided arguments

    """
    def __init__(self, func: Callable, *args, **kwargs):
        _BaseTransformer.__init__(self, *args, **kwargs)
        self.inplace = False
        self.func = func

    def transform(self, df, *args, **kwargs):
        """
        Transforms dataframe according to the saved transformation.
        The user may provide other data via *args, **kwargs arguments.

        :param df: dataframe to be transformed
        :param args: new arguments to be provided to the saved function
        :param kwargs: new keyword arguments to be provided to the saved
        function
        :return: transformed dataframe
        """
        original_pipeline = df.pipeline
        if args == () and kwargs == {}:
            args = self.args
            kwargs = self.kwargs
        with original_pipeline:
            result = self.func(df, *args, **kwargs)
        if result is None:
            result = df
        if df is result:
            result.pipeline = original_pipeline.add(self)
        else:
            result.pipeline = _deepcopy(original_pipeline).add(self)
            result.target = df.target
        return result

    def equals(self, other):
        return _BaseTransformer.equals(self, other) \
               and self.func == other.func

    def __repr__(self):
        return 'Map.' + self.func.__name__ + '(' + self.str_of_args() + ')'
