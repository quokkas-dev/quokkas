from ...utils.string_ops import str_of_args


class Completion:
    """
    Provides an interface for saving the final pipeline transformation,
    i.e. the transformation that makes the dataframe into "not-dataframe"

    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, df, *args, **kwargs):
        if args == ():
            args = self.args
        if kwargs == {}:
            kwargs = self.kwargs
        return self.func(df, *args, **kwargs)

    def __repr__(self):
        return self.func.__name__ + '(' + str_of_args(self.args, self.kwargs) + ')'

    def __eq__(self, other):
        return type(self) == type(other) \
               and self.func.__name__ == self.func.__name__ \
               and self.args == other.args \
               and self.kwargs == other.kwargs
