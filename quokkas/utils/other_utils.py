from __future__ import annotations

from typing import Iterable
import numpy as np

from quokkas.eda.categorical import CategoricalDetector


def sum_not_nones(arr: Iterable) -> int:
    """
    Sums the non-None elements of an array / iterable.

    :param arr: the array / iterable to sum
    :return: the sum of the non-None elements
    """
    return sum(x for x in arr if x is not None)


def create_split(split, i):
    """
    Combines all but one split of an array into one and returns the resulting combined array along with the ignored one.

    :param split: list of splits of an array
    :param i: index of the split to ignore
    :return: the combined array and the ignored split
    """
    return np.concatenate((*split[0:i], *split[(i + 1):])), split[i]


def create_generator(split,
                     n_splits: int,
                     df=None,
                     separate: bool = False,
                     to_numpy: bool = False,
                     return_indices: bool = True):
    """
    Creates a generator for a split of an array that combines all but one split of the array into one and returns the
    resulting combined array along with the ignored one.

    :param split: list of splits of an array
    :param n_splits: number of splits
    :param df: the dataframe to split
    :param separate: if True, the resulting splits will each be returned as features and target column separately.
    If False, the resulting splits will be returned as one dataframe.
    :param to_numpy: if True, 'return_indices' is False and 'separate' is True, the returned dataframes will be
        converted to a numpy array
    :param return_indices: if True, the indices of the dataframes will be returned instead of dataframes
    :return: generator for the split
    """
    if return_indices:
        for i in range(n_splits):
            yield np.concatenate((*split[0:i], *split[(i + 1):])), split[i]
    elif separate:
        for i in range(n_splits):
            yield df.iloc[np.concatenate((*split[0:i], *split[(i + 1):])), :]. \
                      separate(to_numpy=to_numpy), df.iloc[split[i], :].separate(to_numpy=to_numpy)
    elif to_numpy:
        for i in range(n_splits):
            yield df.iloc[np.concatenate((*split[0:i], *split[(i + 1):])), :].to_numpy(), \
                  df.iloc[split[i], :].to_numpy()
    else:
        for i in range(n_splits):
            yield df.iloc[np.concatenate((*split[0:i], *split[(i + 1):])), :], \
                  df.iloc[split[i], :]


def validate_proba(predict_proba: bool | None, df=None, estimator=None, cat_strategy='count',
                   cat_number=CategoricalDetector.DEFAULT_CAT_NUMBER, cat_share=CategoricalDetector.DEFAULT_CAT_SHARE):
    """
    Validates the predict_proba parameter.

    :param predict_proba: bool parameter to be validated
    :param df: dataframe that might contain target, which will be used
    :param cat_strategy: 'count', 'type' or 'count&type'. If
    'count', categorical target will be detected based on the
    number of distinct values (with border being min(cat_share * num_rows,
    cat_number)), if 'type' - based on the type of the column, if
    'count&type' - based on both
    :param cat_number: if 'count' or 'count&type', targets with fewer
    than this number of distinct values will be considered categorical (if
    cat_share * num_rows is larger than cat_number)
    :param cat_share: if 'count' or 'count&type', targets with fewer
    than cat_share * num_rows of distinct values will be considered categorical
    (if cat_number is larger than cat_share * num_rows)
    :param estimator: model, used to check if the model has a predict_proba method
    :return: True if predict_proba is True or None, target (if provided) is categorical, and the model (if provided)
    has a predict_proba method, otherwise False
    """
    CategoricalDetector.validate_strategy(cat_strategy)
    cat_strategy = CategoricalDetector.ALLOWED_STRATEGIES.index(cat_strategy)
    if predict_proba is not None:
        return predict_proba
    predict_proba = (df is not None and len(df.target) > 0) or estimator is not None

    target = df.target.intersection(df.columns)

    if df is not None and len(target) > 0:
        cat_cols = CategoricalDetector.determine_categorical(df, target, cat_strategy, cat_number=cat_number,
                                                             cat_share=cat_share)
        if any(target_col not in cat_cols for target_col in target):
            return False
    if estimator is not None:
        if not hasattr(estimator, 'predict_proba'):
            return False
    return predict_proba


def validate_scorers(scorers, predict_proba):
    """
    Validates the scorers passed to the model.

    :param scorers: the scorers to validate
    :param predict_proba: if True, and the scorers are not provided, the scorers will be set to [log_loss],
    otherwise the scorers will be set to [mean_squared_error]
    :return: the validated scorers and their names (if no scorers were provided)
    """
    if scorers is None:
        if predict_proba:
            from sklearn.metrics import log_loss
            scorers = [log_loss]
            names = ['log_loss']
        else:
            from sklearn.metrics import mean_squared_error
            scorers = [mean_squared_error]
            names = ['mean_squared_error']
    else:
        names = [scorer.__name__ if callable(scorer) else str(scorer) for scorer in scorers]
    return scorers, names
