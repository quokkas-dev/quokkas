from .generic import _BaseTransformer


class Operation(_BaseTransformer):
    """
    The executor class for all native dataframe / ndframe operations.
    Saves the name of the function that was called and its arguments.
    Can then apply the same operations to the test data
    """
    def __init__(self, func_name: str, *args, **kwargs):
        _BaseTransformer.__init__(self, *args, **kwargs)
        self.func_name = func_name

    def make_inplace(self):
        self.inplace = True

    def transform(self, df):
        """
        Transforms the provided data according to the saved function &
        its arguments

        :param df: dataframe to be transformed
        :return: result of the saved function
        """
        method = getattr(df, self.func_name)
        return method(*self.args, **self.kwargs)

    def equals(self, other):
        return _BaseTransformer.equals(self, other) and \
               self.func_name == other.func_name

    def __repr__(self):
        return 'Dataframe.' + self.func_name + '(' + self.str_of_args() + ')'
