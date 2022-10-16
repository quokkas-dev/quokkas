from __future__ import annotations

import inspect
from abc import abstractmethod
import numpy as np
from ..utils.string_ops import str_of_args


class _BaseTransformer:
    """
    Base class for generic transforms.
    Saves arguments and keyword-arguments provided, so that
    the transform can be reused
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.inplace = False

    @abstractmethod
    def transform(self, df):
        pass

    def fit_transform(self, df):
        return self.transform(df)

    def equals(self, other):
        return type(self) == type(other) \
               and self.args == other.args \
               and self.kwargs == other.kwargs \
               and self.inplace == other.inplace

    def str_of_args(self):
        return str_of_args(self.args, self.kwargs)


class _BaseColumnSelector:
    def __init__(self,
                 include: set | list | None = None,
                 exclude: set | list | None = None,
                 include_target: bool = False,
                 auto: bool = True):
        self.auto = auto and (include is None)
        self.include = include if include is None or isinstance(include, set) else set(include)
        self.exclude = exclude if exclude is None or isinstance(exclude, set) else set(exclude)
        self.include_target = include_target

    def _select_columns(self, df):
        """
        Selects columns based on include, exclude and include_target.

        :param df: dataframe from which the columns are selected
        :return: selected columns
        """
        if self.include is not None:
            return self.include

        cols = set(df.columns)

        if self.exclude is not None:
            cols -= self.exclude

        if not self.include_target:
            cols -= df.target if self.include is None else df.target - self.include

        return cols

    def _select_numeric_columns(self, df):
        """
        Selects columns based on include, exclude and include_target.
        If auto, additionally excludes non-numeric columns.

        :param df: dataframe from which the columns are selected
        :return: selected columns
        """
        return self._select_columns(df).intersection(df.select_dtypes(include=np.number).columns) if self.auto and \
            self.include is None else self._select_columns(df)


class _BaseProcessor(_BaseColumnSelector):
    """
    Abstract base class for quokkas implementations of
    preprocessing algorithms. Supports the following arguments:

    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param include_target: if True, the target column will be transformed too
    :param inplace: if True, the transform will be completed inplace
    :param auto: if True, the transform will only be attempted for designated columns
    """

    def __init__(self, inplace: bool = False, auto: bool = True, include: set | list | None = None,
                 exclude: set | list | None = None, include_target: bool = False):
        _BaseColumnSelector.__init__(self, include=include, exclude=exclude, include_target=include_target, auto=auto)
        self.inplace = inplace
        self.fitted = False

    @abstractmethod
    def fit(self, df):
        """
        Fits the processor to the provided data

        :param df: dataframe to be fitted
        """
        pass

    @abstractmethod
    def transform(self, df):
        """
        Transforms the provided data

        :param df: dataframe to be transformed
        :return: transformed dataframe if inplace
        """
        pass

    def fit_transform(self, df):
        """
        Fits the Processor to the provided data and
        transforms data directly afterwards

        :param df: dataframe to be transformed
        :return: transformed dataframe if inplace
        """
        self.fit(df)
        return self.transform(df)

    def __repr__(self):
        return self.__class__.__name__ + '(' + self._get_args_with_values() + ')'

    def _get_args_with_values(self):
        """
        Provides values of arguments with which the
        Processor was initialized.

        :return: string with arguments
        """
        output = []
        args = inspect.getfullargspec(self.__init__)[0][1:]
        for arg in args:
            output.append(f"{arg}={str(getattr(self, arg))}, ")
        return ''.join(output)[:-2]

    @staticmethod
    def check_numeric_type_consistency(df):
        """
        Checks if the numeric blocks have the same type.
        If not, the blockwise transforms won't be attempted,
        as they would make change the types of variables.

        :param df: dataframe to be checked
        :return: True / False if the dataframe typing is
        consistent
        """

        blocks = df._mgr.blocks
        only_type = blocks[0].values.dtype
        for i in range(1, len(blocks)):
            curr_type = blocks[i].values.dtype
            if getattr(curr_type, "_is_numeric", False) and only_type != curr_type:
                return False
        return True

    def equals(self, other):
        return type(self) == type(other) \
               and self.inplace == other.inplace \
               and self.auto == other.auto \
               and self.include == other.include \
               and self.exclude == other.exclude \
               and self.include_target == other.include_target \
               and self.fitted == other.fitted

    def validate_fit(self):
        """
        Validates if the Processor was fitted
        """

        if not self.fitted:
            raise RuntimeError(
                f'The {self.__class__.__name__} was not fitted.\n'
                'Please fit it first before attempting the transformation.')





