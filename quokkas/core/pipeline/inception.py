from copy import deepcopy as _deepcopy
from ...utils.string_ops import str_of_args


class Inception:
    """
    Provides an interface for interacting with functions
    that create dataframes.
    """

    def __init__(self, origin, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.is_callable = callable(origin)
        if self.is_callable:
            self.origin = origin
        else:
            self.origin = origin

    def __call__(self, *args, deep=True, **kwargs):
        if args == ():
            args = self.args
        if kwargs == {}:
            kwargs = self.kwargs

        if self.is_callable:
            return self.origin(*args, **kwargs)
        else:
            from ..frames.dataframe import DataFrame
            if deep:
                return DataFrame(_deepcopy(self.origin), *args, **kwargs)
            else:
                return DataFrame(self.origin, *args, **kwargs)

    def __repr__(self):
        if self.is_callable:
            return self.origin.__name__ + '(' + str_of_args(self.args, self.kwargs) + ')'
        else:
            return self.origin.__class__.__name__ + '(' + str_of_args(self.args, self.kwargs) + ')'

    def __eq__(self, other):
        return type(self) == type(other) \
               and ((self.is_callable and self.origin.__name__ == other.origin.__name__) or (
                    not self.is_callable and self.origin == other.origin)) and self.args == other.args \
               and self.kwargs == other.kwargs
