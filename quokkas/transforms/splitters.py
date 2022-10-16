from __future__ import annotations

from abc import abstractmethod
from typing import Iterable
import numpy as np
from math import floor

from numpy.random import RandomState

from ..utils.other_utils import sum_not_nones
from ..utils.sk_utils import check_random_state, _approximate_mode


class _BaseSplitter:
    """
    Base class for the quokkas implementation of splitters. Supports the following arguments:

    :param inplace: if True, all dataframe transformation within splitter functions happen inplace
    :param sizes: sizes of the resulting splits. If None, the resulting splits will be roughly of equal size.
    Can be provided as absolute or relative sizes.
    :param n_splits: Number of splits. If None, the number of splits will be equal to the length of sizes.
    :param separate: if True, the resulting splits will each be returned as features and target column separately.
    If False, the resulting splits will be returned as one dataframe.
    :param to_numpy: if True, 'return_indices' is False and 'separate' is True, the returned dataframes will be
        converted to a numpy array
    :param return_indices: if True, the indices of the dataframes will be returned instead of dataframes
    :param ordinal_indices: If True and 'return_indices' is True, returns ordinal indices (as those used in iloc),
        otherwise returns indices that correspond to the actual indices of the dataframe (as those used in loc)
    :param random_state: random state used for random splitting (used when kind is 'shuffled' or 'stratified')
    """

    def __init__(self,
                 inplace: bool = False,
                 sizes: Iterable(int) | Iterable(float) | None = None,
                 n_splits: int | None = None,
                 separate: bool = False,
                 to_numpy: bool = False,
                 return_indices: bool = False,
                 ordinal_indices: bool = True,
                 random_state: int | RandomState | None = None,
                 **kwargs):
        self.inplace = inplace
        self.random_state = random_state
        self.sizes = sizes
        self.n_splits = len(sizes) if sizes is not None and n_splits is None else n_splits
        self.separate = separate
        self.to_numpy = to_numpy
        self.return_indices = return_indices
        self.ordinal_indices = ordinal_indices

    def _validate_sizes(self, n_samples: int):
        """
        Validates arguments 'sizes' and 'n_splits'. If not submitted, the resulting splits will be roughly of equal
        sizes.

        :param n_samples: number of samples in the dataframe
        :return: sizes of splits
        """
        sizes = self.sizes if self.sizes is not None else [1 / self.n_splits] * self.n_splits
        strategy = "float" if any(isinstance(size, float) and size not in [0.0, 1.0] for size in sizes) or \
                              all(size is None for size in sizes) else "int"
        if (strategy == "float" and sum_not_nones(sizes) > 1) or sum_not_nones(sizes) > n_samples:
            raise ValueError("Sizes have to be lower or equal to 1 or n_samples.")
        if self.n_splits:
            if len(sizes) not in [self.n_splits, self.n_splits - 1]:
                raise ValueError("Length of sizes has to be equal to n_splits or n_splits - 1: {} is not in [{}, {}]".
                                 format(len(sizes), self.n_splits, self.n_splits - 1))
            if len(sizes) == self.n_splits - 1:
                sizes = list(sizes)
                if strategy == "float":
                    sizes.append(1 - np.sum(sizes))
                else:
                    sizes.append(n_samples - np.sum(sizes))
        if strategy == "float":
            sizes = [floor(size * n_samples) for size in sizes[:-1]]
            sizes.append(n_samples - np.sum(sizes))
        if any(size == 0 for size in sizes):
            raise ValueError("One of the resulting sets will be empty. Please adjust the sizes to avoid this.")
        return sizes

    def split(self, df):
        """
        Validates inputs and then calls the actual splitting function.

        :param df: the dataframe to split
        :return: the resulting splits
        """
        self.sizes = self._validate_sizes(df.shape[0]) if self.sizes is not None else None
        if not self.inplace:
            maybe_df = df.copy(deep=True)
        else:
            maybe_df = df
        with maybe_df.pipeline:
            result = self._split(maybe_df)
        if self.return_indices:
            return result
        elif self.separate:
            return [maybe_df.iloc[_ind, :].separate(to_numpy=self.to_numpy) for _ind in result]
        elif self.to_numpy:
            return [maybe_df.iloc[_ind, :].to_numpy() for _ind in result]
        elif self.inplace:
            result = [maybe_df.iloc[_ind, :] for _ind in result]
            for _df in result:
                _df.is_copy = None
            return result
        else:
            return [maybe_df.iloc[_ind, :].copy(deep=True) for _ind in result]

    @abstractmethod
    def _split(self, maybe_df):
        pass


class StratifiedSplitter(_BaseSplitter):
    """
    Class to perform a stratified split. Supports the following arguments:

    :param inplace: if True, all dataframe transformation within splitter functions happen inplace
    :param sizes: sizes of the resulting splits. If None, the resulting splits will be roughly of equal size.
    Can be provided as absolute or relative sizes.
    :param n_splits: Number of splits. If None, the number of splits will be equal to the length of sizes.
    :param stratify_by: column to stratify by. If not provided, target column is used.
    :param separate: if True, the resulting splits will each be returned as features and target column separately.
    If False, the resulting splits will be returned as one dataframe.
    :param to_numpy: if True, 'return_indices' is False and 'separate' is True, the returned dataframes will be
        converted to a numpy array
    :param return_indices: if True, the indices of the dataframes will be returned instead of dataframes
    :param ordinal_indices: If True and 'return_indices' is True, returns ordinal indices (as those used in iloc),
        otherwise returns indices that correspond to the actual indices of the dataframe (as those used in loc)
    :param random_state: random state used for random splitting
    """
    def __init__(self,
                 inplace: bool = False,
                 sizes: Iterable(int) | Iterable(float) | None = None,
                 n_splits: int | None = None,
                 stratify_by: str | None = None,
                 separate: bool = False,
                 to_numpy: bool = False,
                 return_indices: bool = False,
                 ordinal_indices: bool = True,
                 random_state: int | RandomState | None = None,
                 **kwargs):
        _BaseSplitter.__init__(self, inplace=inplace, random_state=random_state, sizes=sizes, n_splits=n_splits,
                               ordinal_indices=ordinal_indices, return_indices=return_indices, separate=separate,
                               to_numpy=to_numpy, **kwargs)
        self.stratify_by = stratify_by

    def _split(self, df):
        """
        Iterates through splits calling _stratify_split function, which returns stratified split indices (ordinal)
        that are then saved and are not considered in the following iterations.

        :param df: dataframe to split
        :return: list of ordinal indices for each split
        """
        self.sizes = self._validate_sizes(df.shape[0]) if self.sizes is None else self.sizes
        stratify_by = df.target if not self.stratify_by else self.stratify_by
        if (hasattr(stratify_by, "__len__") and len(stratify_by) > 1) or \
                (hasattr(stratify_by, "shape") and (
                        stratify_by.shape.dim > 1 or stratify_by.shape[0] > 1)):
            raise ValueError("stratify_by cannot be a set, list or tuple with more than one element.")
        if (hasattr(stratify_by, "__len__") and len(stratify_by) == 0) or \
                (hasattr(stratify_by, "shape") and (stratify_by.shape[0] == 0)):
            raise ValueError("stratify_by cannot be an empty set, list or tuple.")
        if hasattr(stratify_by, "__iter__"):
            stratify_by = next(iter(stratify_by))
        if stratify_by not in df.columns:
            raise ValueError("Stratification column '" + str(stratify_by) + "' not found in dataframe.")
        result = []
        ind = df.index.to_numpy() if not self.ordinal_indices else np.arange(df.shape[0])
        for size in self.sizes[:-1]:
            to_stratify = df.loc[ind, stratify_by] if not self.ordinal_indices else df.iloc[ind][stratify_by]
            ind_res = self._stratify_split(to_stratify, size)
            out_ind = ind[ind_res]
            result.append(out_ind)
            ind = np.setdiff1d(ind, out_ind)
        result.append(ind)
        return result

    def _stratify_split(self, X, n_draw: int):
        """
        Draws samples from the provided array in a stratified manner, i.e. X is split into classes based on its unique
        values, and the number of samples drawn from each class is proportional to the size of the class.

        :param X: column to stratify by
        :param n_draw: Number of samples to draw from X
        :return: indices of the drawn samples
        """
        classes, y_indices = np.unique(X, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)
        if np.min(class_counts) < 2:
            raise ValueError(
                "The least populated class of the column to stratify by has only 1"
                " member, which is too few. The minimum"
                " number of groups for any class cannot"
                " be less than 2."
            )

        if n_draw < n_classes:
            raise ValueError(
                "The size = %d should be greater or "
                "equal to the number of classes = %d" % (n_draw, n_classes)
            )

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        rng = check_random_state(self.random_state)

        n_i = _approximate_mode(class_counts, n_draw, rng)

        train = []

        for i in range(n_classes):
            permutation = rng.permutation(class_counts[i])
            perm_indices_class_i = class_indices[i].take(permutation, mode="clip")
            train.extend(perm_indices_class_i[: n_i[i]])

        train = rng.permutation(train)

        return train


class SortedSplitter(_BaseSplitter):
    """
    Class to perform a sorted split. Supports the following arguments:

    :param inplace: if True, all dataframe transformation within splitter functions happen inplace
    :param sizes: sizes of the resulting splits. If None, the resulting splits will be roughly of equal size.
    Can be provided as absolute or relative sizes.
    :param n_splits: Number of splits. If None, the number of splits will be equal to the length of sizes.
    :param sort_by: column to sort by. If not provided, target column is used.
    :param ascending: if True, the dataframe will be sorted in ascending order, otherwise in descending order.
    :param separate: if True, the resulting splits will each be returned as features and target column separately.
    If False, the resulting splits will be returned as one dataframe.
    :param to_numpy: if True, 'return_indices' is False and 'separate' is True, the returned dataframes will be
        converted to a numpy array
    :param return_indices: if True, the indices of the dataframes will be returned instead of dataframes
    :param ordinal_indices: If True and 'return_indices' is True, returns ordinal indices (as those used in iloc),
        otherwise returns indices that correspond to the actual indices of the dataframe (as those used in loc)
    """
    def __init__(self,
                 inplace: bool = False,
                 sizes: Iterable(int) | Iterable(float) | None = None,
                 n_splits: int | None = None,
                 sort_by: str | None = None,
                 ascending: bool = True,
                 separate: bool = False,
                 to_numpy: bool = False,
                 return_indices: bool = False,
                 ordinal_indices: bool = True,
                 **kwargs):
        _BaseSplitter.__init__(self, inplace=inplace, sizes=sizes,
                               return_indices=return_indices, n_splits=n_splits, ordinal_indices=ordinal_indices,
                               separate=separate, to_numpy=to_numpy, **kwargs)
        self.sort_by = sort_by
        self.ascending = ascending

    def _split(self, df):
        """
        Performs a sorted split by sorting the dataframe by the provided column first and then performing a
        sequential split.

        :param df: dataframe to split
        :return: list of indices of the resulting splits
        """
        self.sizes = self._validate_sizes(df.shape[0]) if self.sizes is None else self.sizes
        if self.sort_by is not None and self.sort_by not in df.columns:
            raise ValueError("Sort column '" + self.sort_by + "' not found in dataframe.")
        sort_by = df.target if self.sort_by is None else self.sort_by
        if sort_by is None:
            raise ValueError("Sort column cannot be None.")
        result = []
        df.sort_values(by=self.sort_by, ascending=self.ascending, inplace=True)
        ind = df.index.to_numpy() if not self.ordinal_indices else np.arange(df.shape[0])
        start_ind = 0
        for i in range(self.n_splits):
            stop_ind = start_ind + self.sizes[i]
            result.append(ind[start_ind:stop_ind])
            start_ind = stop_ind
        return result


class ShuffledSplitter(_BaseSplitter):
    """
    Class to perform a shuffled split. Supports the following arguments:

    :param inplace: if True, all dataframe transformation within splitter functions happen inplace
    :param sizes: sizes of the resulting splits. If None, the resulting splits will be roughly of equal size.
    Can be provided as absolute or relative sizes.
    :param n_splits: Number of splits. If None, the number of splits will be equal to the length of sizes.
    :param separate: if True, the resulting splits will each be returned as features and target column separately.
    If False, the resulting splits will be returned as one dataframe.
    :param to_numpy: if True, 'return_indices' is False and 'separate' is True, the returned dataframes will be
        converted to a numpy array
    :param return_indices: if True, the indices of the dataframes will be returned instead of dataframes
    :param ordinal_indices: If True and 'return_indices' is True, returns ordinal indices (as those used in iloc),
        otherwise returns indices that correspond to the actual indices of the dataframe (as those used in loc)
    :param random_state: random state used for random splitting
    """
    def _split(self, df):
        """
        Performs a shuffled split by picking random indices for each split iteratively and then excluding the picked
        indices in the next iterations.

        :param df: dataframe to split
        :return: list of indices of the resulting splits
        """
        rng = check_random_state(self.random_state)
        ind = df.index.to_numpy() if not self.ordinal_indices else np.arange(df.shape[0])
        if self.sizes is None:
            if self.n_splits is None:
                raise ValueError("n_splits has to be set if sizes is None.")
            return np.array_split(rng.permutation(ind), self.n_splits, axis=0)
        result = []
        for i in range(self.n_splits):
            _ind = rng.choice(ind, size=self.sizes[i], replace=False)
            result.append(_ind)
            ind = np.setdiff1d(ind, _ind)
        return result


class SequentialSplitter(_BaseSplitter):
    """
    Class to perform a sequential split. Example: when 'sizes' is (0.8, 0.2), the first split will contain the first 80%
    of the rows in the dataframe, and the second split the last 20% of the rows. Supports the following arguments:

    :param inplace: if True, all dataframe transformation within splitter functions happen inplace
    :param sizes: sizes of the resulting splits. If None, the resulting splits will be roughly of equal size.
    Can be provided as absolute or relative sizes.
    :param n_splits: Number of splits. If None, the number of splits will be equal to the length of sizes.
    :param separate: if True, the resulting splits will each be returned as features and target column separately.
    If False, the resulting splits will be returned as one dataframe.
    :param to_numpy: if True, 'return_indices' is False and 'separate' is True, the returned dataframes will be
        converted to a numpy array
    :param return_indices: if True, the indices of the dataframes will be returned instead of dataframes
    :param ordinal_indices: If True and 'return_indices' is True, returns ordinal indices (as those used in iloc),
        otherwise returns indices that correspond to the actual indices of the dataframe (as those used in loc)
    """
    def _split(self, df):
        """
        Performs a sequential split.

        :param df: dataframe to split
        :return: list of indices of the resulting splits
        """
        ind = df.index.to_numpy() if not self.ordinal_indices else np.arange(df.shape[0])
        if self.sizes is None:
            if self.n_splits is None:
                raise ValueError("n_splits has to be set if sizes is None.")
            return np.array_split(ind, self.n_splits, axis=0)
        result = []
        start_ind = 0
        for i in range(self.n_splits):
            stop_ind = start_ind + self.sizes[i]
            result.append(ind[start_ind:stop_ind])
            start_ind = stop_ind
        return result
