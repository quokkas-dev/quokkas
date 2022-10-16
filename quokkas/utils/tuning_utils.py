from __future__ import annotations

import inspect
import numbers

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

from .pd_utils.pd_typing import RandomState
from .sk_utils import _score

ALLOWED_KINDS = {'grid', 'randomized'}


def prepare_search_params(kind: str, params: dict, n_iter: int | None, random_state: int | RandomState | None):
    if kind not in ALLOWED_KINDS:
        raise ValueError(f'Unknown search kind passed: '
                         f'expected one of {ALLOWED_KINDS}, got {str(kind)}')

    if kind == 'grid':
        return ParameterGrid(params)
    else:
        return ParameterSampler(params, n_iter, random_state=random_state)


def clone_model(estimator, params):
    """
    Creates an unfitted copy of an estimator with the same parameters.

    :param estimator: estimator to be cloned
    :return: clone of the estimator
    """
    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone_model(e, params) for e in estimator])
    klass = estimator.__class__
    if isinstance(estimator, type) or not hasattr(estimator, 'fit'):
        return estimator(**params)

    if not hasattr(estimator, 'get_params'):
        return klass(**params)
    new_object_params = estimator.get_params(deep=True)
    new_object_params.update(params)
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


def separate_params(kind, params, fit_params, n_iter, random_state):
    if kind not in ALLOWED_KINDS:
        raise ValueError(f'Unknown search kind passed: '
                         f'expected one of {ALLOWED_KINDS}, got {str(kind)}')

    if fit_params is not None:
        params.update({f'&{key}': value for key, value in fit_params.items()})

    init_params = []
    fit_params = []

    params = ParameterGrid(params) if kind == 'grid' else ParameterSampler(params, n_iter, random_state=random_state)

    # should have been one comprehension, but thx python
    for param in params:
        init_tmp = {}
        fit_tmp = {}

        for key, value in param.items():
            if key[0] == '&':
                fit_tmp[key[1:]] = value
            else:
                init_tmp[key] = value
        init_params.append(init_tmp)
        fit_params.append(fit_tmp)

    return init_params, fit_params


def pretty_results(params, scorer_names, test_scores, train_scores=None, **kwargs):
    from ..core.frames.dataframe import DataFrame
    test_score = test_scores if isinstance(test_scores, np.ndarray) else np.array(test_scores)

    output = DataFrame(params)
    if len(test_score.shape) == 3:
        for i, score in enumerate(scorer_names):
            output[str(score) + '_test'] = np.nanmean(test_score[:, :, i], axis=1)
    else:
        for i, score in enumerate(scorer_names):
            output[str(score) + '_test'] = test_score[:, i]

    if train_scores is None:
        return output

    train_score = train_scores if isinstance(train_scores, np.ndarray) else np.array(train_scores)

    if len(train_score.shape) == 3:
        for i, score in enumerate(scorer_names):
            output[str(score) + '_train'] = np.nanmean(train_score[:, :, i], axis=1)
    else:
        for i, score in enumerate(scorer_names):
            output[str(score) + '_train'] = train_score[:, i]

    return output


def _fit_and_score_separated(
        estimator,
        X_train,
        y_train,
        X_test,
        y_test,
        scorers,
        fit_params: dict | None = None,
        return_train_score: bool = False,
        error_score=np.nan,
        proba: bool = False
):
    """
    Fits and scores an estimator for a given split and returns a dict containing the summary.

    :param estimator: estimator object used to fit the data and make predictions
    :param X_train: feature data to be fitted
    :param y_train: train labels
    :param X_test: feature data to be tested
    :param y_test: test labels
    :param scorers: scoring function(s) as callable or list of callables to use for cross-validation. Each callable
    function has to take two vectors as inputs and return a single value
    :param fit_params: parameters to pass to the estimator's fit method
    :param return_train_score: if True, train scores will be returned as part the resulting dictionary
    :param error_score: value to assign to the score if an error occurs in estimator fitting
    :param proba: if True, the estimator will be fitted and predictions made using the predict_proba method
    :return: a dictionary containing the summary of the fit and score
    """

    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            test_scores = np.repeat(error_score, len(scorers))
            if return_train_score:
                train_scores = np.repeat(error_score, len(scorers))
    else:
        test_scores, _ = _score(estimator, X_test, y_test, scorers, error_score, proba)
        if return_train_score:
            train_scores, _ = _score(estimator, X_train, y_train, scorers, error_score, proba)

    return {'test_scores': test_scores, 'train_scores': train_scores} if return_train_score else {
        'test_scores': test_scores}
