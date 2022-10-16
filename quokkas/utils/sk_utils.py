"""
This code was originally taken from scikit-learn:
https://scikit-learn.org/stable/
Some of it may have been modified.

The use of this code is allowed under BSD license. scikit-learn license can
be found below:

BSD 3-Clause License

Copyright (c) 2007-2021 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Please also refer to the "LICENSES" directory of this repository.
"""
import copy
import numbers
import math
import warnings
from collections import Counter
from contextlib import suppress
from traceback import format_exc
from typing import NamedTuple
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.exceptions import FitFailedWarning


def is_scalar_nan(x):
    """
    Test if x is NaN.

    :param x: The value to test.
    :return: True if x is NaN, False otherwise.
    """
    return isinstance(x, numbers.Real) and math.isnan(x)


class _nandict(dict):
    """
    This class represents a dictionary with a key 'nan_value' that is used to represent the NaN value.
    When a key is accessed that is missing from the dictionary, the value associated with the key 'nan_value' is
    returned.
    """

    def __init__(self, mapping):
        super().__init__(mapping)
        for key, value in mapping.items():
            if is_scalar_nan(key):
                self.nan_value = value
                break

    def __missing__(self, key):
        """
        This method is called when a key is accessed that is missing from the dictionary.
        :param key: The key that is missing from the dictionary.
        :return: The value associated with the key 'nan_value'.
        """
        if hasattr(self, "nan_value") and is_scalar_nan(key):
            return self.nan_value
        raise KeyError(key)


def check_sorted(categories, dtype):
    """
    Checks if the categories are sorted in case of numeric categories. If there is a NaN category, it is supposed to be
    the last element. If not, or if the categories are not sorted, raises an error.

    :param categories: the categories to check
    :param dtype: the dtype of the categories
    """
    cats = np.array(categories, dtype=dtype)
    if dtype.kind not in "OUS":
        sorted_cats = np.sort(cats)
        # if there are nans, nan should be the last element
        stop_idx = -1 if np.isnan(sorted_cats[-1]) else None
        if np.any(sorted_cats[:stop_idx] != cats[:stop_idx]) or (
                np.isnan(sorted_cats[-1]) and not np.isnan(sorted_cats[-1])
        ):
            raise ValueError("Unsorted categories are not supported for numerical categories")


def object_diff(values, known_values):
    """
    Determines the difference between two sets containing unique values of 'values' and 'known_values' respectively.
    Handles missing values by considering None and all other NaNs as two separate categories (unique values).

    :param values: the set to compare with 'known_values'
    :param known_values: the set to compare with 'values'
    :return: a list containing the unique values in 'values' that are not in 'known_values', including missing values
    """
    values_set = set(values)
    values_set, missing_in_values = _extract_missing(values_set)

    uniques_set = set(known_values)
    uniques_set, missing_in_uniques = _extract_missing(uniques_set)
    diff = values_set - uniques_set

    nan_in_diff = missing_in_values.nan and not missing_in_uniques.nan
    none_in_diff = missing_in_values.none and not missing_in_uniques.none

    diff = list(diff)
    if none_in_diff:
        diff.append(None)
    if nan_in_diff:
        diff.append(np.nan)
    return diff


def _extract_missing(values):
    """
    Extracts missing values from the provided set.

    :param values: the set to extract missing values from
    :return: the original set with missing values extracted, a MissingValues object with information on missing values
    """
    missing_values_set = {
        value for value in values if value is None or is_scalar_nan(value)
    }
    if not missing_values_set:
        return values, MissingValues(nan=False, none=False)

    if None in missing_values_set:
        if len(missing_values_set) == 1:
            output_missing_values = MissingValues(nan=False, none=True)
        else:
            # If there is more than one missing value, then it has to be
            # float('nan') or np.nan
            output_missing_values = MissingValues(nan=True, none=True)
    else:
        output_missing_values = MissingValues(nan=True, none=False)

    # create set without the missing values
    output = values - missing_values_set
    return output, output_missing_values


class MissingValues(NamedTuple):
    """
    Class containing information on missing data.

    :param nan: True if there is a NaN value, False otherwise.
    :param none: True if there is a None value, False otherwise.
    """
    nan: bool
    none: bool

    def to_list(self):
        """
        Returns a list containing types of missing values.

        :return: a list with None first if present
        """
        output = []
        if self.none:
            output.append(None)
        if self.nan:
            output.append(np.nan)
        return output


def _get_ordered_idx(mask_missing_values, imputation_order, skip_complete, random_state):
    """
    Determines the order in which the features will be updated.

    :param mask_missing_values:
    :param imputation_order: strategy to determine the order. Possible values:
    'ascending': From features with fewest missing values to most.
    'descending': From features with most missing values to fewest
    'roman': Left to right
    'arabic': Right to left
    'random': A random order for each round
    :param skip_complete: if True, skips features with no missing values
    :param random_state: random state for the random order
    :return: the order in which to impute the features as a numpy array of indices
    """
    frac_of_missing_values = mask_missing_values.mean(axis=0)
    if skip_complete:
        missing_values_idx = np.flatnonzero(frac_of_missing_values)
    else:
        missing_values_idx = np.arange(np.shape(frac_of_missing_values)[0])
    if imputation_order == "roman":
        ordered_idx = missing_values_idx
    elif imputation_order == "arabic":
        ordered_idx = missing_values_idx[::-1]
    elif imputation_order == "ascending":
        n = len(frac_of_missing_values) - len(missing_values_idx)
        ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:]
    elif imputation_order == "descending":
        n = len(frac_of_missing_values) - len(missing_values_idx)
        ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:][::-1]
    elif imputation_order == "random":
        ordered_idx = missing_values_idx
        random_state.shuffle(ordered_idx)
    else:
        raise ValueError(
            "Got an invalid imputation order: '{0}'. It must "
            "be one of the following: 'roman', 'arabic', "
            "'ascending', 'descending', or "
            "'random'.".format(imputation_order)
        )
    return ordered_idx


def _most_frequent(array):
    """
    Computes the most frequent value in a 1d array.

    :param array: the array to compute the most frequent value for
    :return: the most frequent value
    """
    # Compute the most frequent value in array only
    if array.size > 0:

        if array.dtype == object:
            # scipy.stats.mode is slow with object dtype array.
            # Python Counter is more efficient
            counter = Counter(array)
            most_frequent_count = counter.most_common(1)[0][1]
            # tie breaking similarly to scipy.stats.mode
            try:  # min won't work for strings
                most_frequent_value = min(
                    value
                    for value, count in counter.items()
                    if count == most_frequent_count
                )
            except TypeError:
                most_frequent_value = [value for value, count in counter.items() if count == most_frequent_count][0]
        else:
            most_frequent_value = stats.mode(array, nan_policy='omit', keepdims=False)[0]
    else:
        most_frequent_value = 0

    return most_frequent_value


def _unique_python(values, return_counts):
    """

    :param values:
    :param return_counts:
    :return:
    """
    try:
        uniques_set = set(values)
        uniques_set, missing_values = _extract_missing(uniques_set)

        uniques = sorted(uniques_set)
        uniques.extend(missing_values.to_list())
        uniques = np.array(uniques, dtype=values.dtype)
    except TypeError:
        types = sorted(t.__qualname__ for t in set(type(v) for v in values))
        raise TypeError(
            "Encoders require their input to be uniformly "
            f"strings or numbers. Got {types}"
        )
    ret = (uniques,)

    if return_counts:
        ret += (_get_counts(values, uniques),)

    return ret[0] if len(ret) == 1 else ret


def _get_counts(values, uniques):
    """Get the count of each of the `uniques` in `values`.

    The counts will use the order passed in by `uniques`. For non-object dtypes,
    `uniques` is assumed to be sorted and `np.nan` is at the end.
    """
    if values.dtype.kind in "OU":
        counter = _NaNCounter(values)
        output = np.zeros(len(uniques), dtype=np.int64)
        for i, item in enumerate(uniques):
            with suppress(KeyError):
                output[i] = counter[item]
        return output

    unique_values, counts = _unique_np(values, return_counts=True)

    # Recorder unique_values based on input: `uniques`
    uniques_in_values = np.isin(uniques, unique_values, assume_unique=True)
    if np.isnan(unique_values[-1]) and np.isnan(uniques[-1]):
        uniques_in_values[-1] = True

    unique_valid_indices = np.searchsorted(unique_values, uniques[uniques_in_values])
    output = np.zeros_like(uniques, dtype=np.int64)
    output[uniques_in_values] = counts[unique_valid_indices]
    return output


class _NaNCounter(Counter):
    """Counter with support for nan values."""

    def __init__(self, items):
        super().__init__(self._generate_items(items))

    def _generate_items(self, items):
        """Generate items without nans. Stores the nan counts separately."""
        for item in items:
            if not is_scalar_nan(item):
                yield item
                continue
            if not hasattr(self, "nan_count"):
                self.nan_count = 0
            self.nan_count += 1

    def __missing__(self, key):
        if hasattr(self, "nan_count") and is_scalar_nan(key):
            return self.nan_count
        raise KeyError(key)


def _unique_np(values, return_inverse=False, return_counts=False):
    """
    Finds the unique elements of an array, handles all NaNs as one value.
    :param values: values to find unique elements of
    :param return_inverse: if True, also returns the indices of the unique elements
    :param return_counts: if True, also returns the counts of each unique element
    :return: unique elements, inverse indices, counts
    """
    uniques = np.unique(
        values, return_inverse=return_inverse, return_counts=return_counts
    )

    inverse, counts = None, None

    if return_counts:
        *uniques, counts = uniques

    if return_inverse:
        *uniques, inverse = uniques

    if return_counts or return_inverse:
        uniques = uniques[0]

    # np.unique will have duplicate missing values at the end of `uniques`
    # here we clip the nans and remove it from uniques
    if uniques.size and is_scalar_nan(uniques[-1]):
        nan_idx = np.searchsorted(uniques, np.nan)
        uniques = uniques[: nan_idx + 1]
        if return_inverse:
            inverse[inverse > nan_idx] = nan_idx

        if return_counts:
            counts[nan_idx] = np.sum(counts[nan_idx:])
            counts = counts[: nan_idx + 1]

    ret = (uniques,)

    if return_inverse:
        ret += (inverse,)

    if return_counts:
        ret += (counts,)

    return ret[0] if len(ret) == 1 else ret


def _get_dense_mask(X, value_to_mask):
    """
    Return a mask for the values equal to value_to_mask. Can be used for missing values.

    :param X: array to compute the mask for
    :param value_to_mask: value to mask
    :return: mask
    """
    if value_to_mask is pd.NA:
        return pd.isna(X)
    if is_scalar_nan(value_to_mask):
        if X.dtype.kind == "f":
            Xt = np.isnan(X)
        elif X.dtype.kind in ("i", "u"):
            # can't have NaNs in integer array.
            Xt = np.zeros(X.shape, dtype=bool)
        else:
            # np.isnan does not work on object dtypes.
            Xt = (X != X)
    else:
        Xt = X == value_to_mask
    return Xt


def _get_weights(dist, weights):
    """
    Computes the weights from an array of distances and the 'weights' parameter.

    :param dist: input distances
    :param weights: the kind of weighting used. Possible values: 'uniform', 'distance' or a callable
    :return: None if weights == 'uniform', else an array of the same shape as 'dist' containing the weights
    """
    if weights in (None, "uniform"):
        return None
    elif weights == "distance":
        # if user attempts to classify a point that was zero distance from one
        # or more training points, those training points are weighted as 1.0
        # and the other points as 0.0
        if dist.dtype is np.dtype(object):
            for point_dist_i, point_dist in enumerate(dist):
                # check if point_dist is iterable
                # (ex: RadiusNeighborClassifier.predict may set an element of
                # dist to 1e-6 to represent an 'outlier')
                if hasattr(point_dist, "__contains__") and 0.0 in point_dist:
                    dist[point_dist_i] = point_dist == 0.0
                else:
                    dist[point_dist_i] = 1.0 / point_dist
        else:
            with np.errstate(divide="ignore"):
                dist = 1.0 / dist
            inf_mask = np.isinf(dist)
            inf_row = np.any(inf_mask, axis=1)
            dist[inf_row] = inf_mask[inf_row]
        return dist
    elif callable(weights):
        return weights(dist)
    else:
        raise ValueError(
            "weights not recognized: should be 'uniform', "
            "'distance', or a callable function"
        )


def _approximate_mode(class_counts, n_draws, rng):
    """
    Computes an approximation to the mode of the multivariate hypergeometric. Shouldn't be off by more than one.

    :param class_counts: number of samples per class
    :param n_draws: number of samples to draw from the overall population
    :param rng: random state
    :return: array with the number of samples to draw from each class
    """
    rng = check_random_state(rng)
    # this computes a bad approximation to the mode of the
    # multivariate hypergeometric given by class_counts and n_draws
    continuous = class_counts / class_counts.sum() * n_draws
    # floored means we don't overshoot n_samples, but probably undershoot
    floored = np.floor(continuous)
    # we add samples according to how much "left over" probability
    # they had, until we arrive at n_samples
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        # add according to remainder, but break ties
        # randomly to avoid biases
        for value in values:
            (inds,) = np.where(remainder == value)
            # if we need_to_add less than what's in inds
            # we draw randomly from them.
            # if we need to add more, we add them all and
            # go to the next value
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(int)


def check_random_state(seed):
    """
    Turns seed into a np.random.RandomState instance.

    :param seed: None, int or instance of RandomState to generate RandomState object from
    :return: np.random.RandomState object based on 'seed'
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


def _warn_or_raise_about_fit_failures(results, error_score):
    """
    Warns or raises an error if one or multiple fits failed.

    :param results: dict containing the results of fit(s)
    :param error_score: the score that will be used for the failed fit(s)
    """
    fit_errors = [
        result["fit_error"] for result in results if result["fit_error"] is not None
    ]
    if fit_errors:
        num_failed_fits = len(fit_errors)
        num_fits = len(results)
        fit_errors_counter = Counter(fit_errors)
        delimiter = "-" * 80 + "\n"
        fit_errors_summary = "\n".join(
            f"{delimiter}{n} fits failed with the following error:\n{error}"
            for error, n in fit_errors_counter.items()
        )

        if num_failed_fits == num_fits:
            all_fits_failed_message = (
                f"\nAll the {num_fits} fits failed.\n"
                "It is very likely that your model is misconfigured.\n"
                "You can try to debug the error by setting error_score='raise'.\n\n"
                f"Below are more details about the failures:\n{fit_errors_summary}"
            )
            raise ValueError(all_fits_failed_message)

        else:
            some_fits_failed_message = (
                f"\n{num_failed_fits} fits failed out of a total of {num_fits}.\n"
                "The score on these train-test partitions for these parameters"
                f" will be set to {error_score}.\n"
                "If these failures are not expected, you can try to debug them "
                "by setting error_score='raise'.\n\n"
                f"Below are more details about the failures:\n{fit_errors_summary}"
            )
            warnings.warn(some_fits_failed_message, FitFailedWarning)


def _insert_error_scores(results, error_score):
    """
    Inserts 'error_score' in `results` fot fit(s) that failed. Only handles multimetric scores.

    :param results: dict containing the results of fit(s)
    :param error_score: the score that will be used for the failed fit(s)
    """
    successful_score = None
    failed_indices = []
    for i, result in enumerate(results):
        if result["fit_error"] is not None:
            failed_indices.append(i)
        elif successful_score is None:
            successful_score = result["test_scores"]

    if isinstance(successful_score, dict):
        formatted_error = {name: error_score for name in successful_score}
        for i in failed_indices:
            results[i]["test_scores"] = formatted_error.copy()
            if "train_scores" in results[i]:
                results[i]["train_scores"] = formatted_error.copy()


def _aggregate_score_dicts(scores):
    """
    Aggregates the scores into one dict.

    :param scores: list of dicts with scores for each fit
    :return: dict with the aggregated scores
    """
    return {
        key: np.asarray([score[key] for score in scores])
        if isinstance(scores[0][key], numbers.Number)
        else [score[key] for score in scores]
        for key in scores[0]
    }


def _normalize_score_results(scores, scaler_score_key="score"):
    """
    Creates a scoring dictionary for multimetric (list of dicts) and single-metric (scalar) scores.

    :param scores: list of dicts with scores for each fit or a scalar
    :param scaler_score_key: key in the dict that contains the scalar score
    :return: dict with aggregated scores
    """
    if isinstance(scores[0], dict):
        # multimetric scoring
        return _aggregate_score_dicts(scores)
    # scaler
    return {scaler_score_key: scores}


def _score(estimator, X_test, y_test, scorer, error_score="raise", predict_proba=False):
    """
    Computes the score(s) of an estimator on the provided test set.

    :param estimator: estimator instance
    :param X_test: test set features
    :param y_test: test set target
    :param scorer: a single- or multi-metric scorer
    :param error_score: the score used for the failed fits
    :param predict_proba: whether to predict probabilities
    :return: scores as an array of floats or a float scalar, predictions as an array of floats
    """
    scores = np.repeat(error_score, len(scorer))
    y_pred = None
    try:
        if predict_proba:
            y_pred = estimator.predict_proba(X_test) if hasattr(estimator, 'predict_proba') else estimator.predict(X_test)
        else:
            y_pred = estimator.predict(X_test)
        if not isinstance(scorer, list):
            scorer = [scorer]
        for i, scor in enumerate(scorer):
            try:
                scores[i] = scor(y_test, y_pred)
            except Exception:
                if error_score == "raise":
                    raise
                warnings.warn(
                    "Scoring failed. The score on this train-test partition for "
                    f"these parameters will be set to {error_score}. Details: \n"
                    f"{format_exc()}",
                    UserWarning,
                )
    except Exception:
        if error_score == "raise":
            raise
        warnings.warn(
            "Scoring failed. The score on this train-test partition for "
            f"these parameters will be set to {error_score}. Details: \n"
            f"{format_exc()}",
            UserWarning,
        )
    return scores, y_pred


def clone(estimator):
    """
    Creates an unfitted copy of an estimator with the same parameters.

    :param estimator: estimator to be cloned
    :return: clone of the estimator
    """
    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e) for e in estimator])
    elif not hasattr(estimator, "get_params") or isinstance(estimator, type):
        return copy.deepcopy(estimator)

    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError(
                "Cannot clone object %s, as the constructor "
                "either does not set or modifies parameter %s" % (estimator, name)
            )
    return new_object
