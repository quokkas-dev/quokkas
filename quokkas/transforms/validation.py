from __future__ import annotations

import numbers
import time
from traceback import format_exc

from joblib import Parallel, delayed, logger
import numpy as np
from numpy.random import RandomState

from ..core.pipeline.lock import Lock
from ..utils.other_utils import validate_scorers, validate_proba
from ..utils.sk_utils import _warn_or_raise_about_fit_failures, _score, _normalize_score_results, _insert_error_scores, \
    _aggregate_score_dicts, clone

from .generic import _BaseProcessor


class CrossValidator(_BaseProcessor):
    """
    Performs cross-validation on the provided estimator using metrics provided in the 'scoring' parameter.

    :param include: if provided, the transform will be applied only to these columns
    :param exclude: if provided, the transform will not be applied to these columns
    :param auto: if True, the transform will only be attempted for numeric columns
    :param target: target column, used if the dataframe has no target
    :param estimator: estimator to use for cross-validation
    :param cv: number of folds for cross-validation
    :param split_kind: kind of split to use for cross-validation. Possible values:
    - 'shuffled' (equivalent to ShuffledSplitter),
    - 'sequential' (equivalent to SequentialSplitter),
    - 'sorted' (equivalent to SortedSplitter), supports arguments 'sort_by' (column to be used for sorting;
    if provided, splits will not be generated randomly, instead they will be generated according to the order of
    the column) and 'ascending' (when sort_by is provided, is used to reverse the sorting order),
    - 'stratified' (equivalent to StratifiedSplitter), supports argument 'stratify_by' (if provided, the dataframes
    will be stratified according to the values of this column, and of the target column otherwise)
    :param stratify_by: column to stratify by when split_kind is 'stratified'. If not provided, target column is used.
    :param sort_by: column to be used for sorting when split_kind is 'sorted'
    :param ascending: when split_kind is 'sorted', the dataframe will be sorted in ascending order if True, otherwise
    in descending order
    :param scorers: scoring function(s) as callable or list of callables to use for cross-validation. Each callable
    function has to take two vectors as inputs and return a single value
    :param error_score: value to assign to the score if an error occurs. If 'raise', the error is raised
    :param fit_params: parameters to pass to the fit method of the estimator
    :param proba: if True, predict_proba method will be used for prediction instead of predict
    :param pre_dispatch: number of jobs to dispatch to workers. Can be used to control the speed and memory consumption
    in the parallel processing. Possible values:
    - None, in which case all the jobs are immediately created and spawned
    - int, denoting the number of total jobs that are spawned
    - str, an expression as a function of n_jobs, such as '2*n_jobs'
    :param n_jobs: number of jobs to run in parallel. If -1, all CPUs are used. If 1, no parallelization is used
    :param verbose: verbosity level
    :param return_train_score: if True, the training scores will be added to the returned dictionary
    :param return_estimator: if True, the estimator will be added to the returned dictionary
    :param return_test_indices: if True, predictions will be added to the returned dictionary
    :param return_predictions: if True, predictions will be added to the returned dictionary
    :param random_state: random state to use for shuffling and stratification
    :return: a dictionary with the following keys:
    - 'test_scores': an array with test scores for each scoring function
    - 'train_scores': an array with training scores for each scoring function, if return_train_score is True
    - 'estimator': the estimator used for cross-validation, if return_estimator is True
    - 'test_indices': an array with the ordinal indices of the test set for each fold
    - 'predictions': an array with predictions for each scoring function, if return_predictions is True
    - 'fit_time': time spent fitting the estimator on the train set for each cv split
    - 'score_time': time spent scoring the estimator on the test set for each cv split
    - 'scorer_names': list of scorer names
    """
    def __init__(self,
                 include: set | list | None = (),
                 exclude: set | list | None = (),
                 auto: bool = True,
                 target: str | None = None,
                 estimator=None,
                 cv: int = 5,
                 split_kind: str = 'shuffled',
                 sort_by: str | None = None,
                 ascending: bool = True,
                 stratify_by: str | None = None,
                 scorers=None,
                 error_score: float | int | np.nan | None = np.nan,
                 fit_params: dict | None = None,
                 proba: bool = False,
                 pre_dispatch: int | str = "2*n_jobs",
                 n_jobs: int = None,
                 verbose: int = 0,
                 return_train_score: bool = False,
                 return_estimator: bool = False,
                 return_test_indices: bool = False,
                 return_predictions: bool = False,
                 random_state: int | RandomState | None = None):
        _BaseProcessor.__init__(self, inplace=False, auto=auto, include=include, exclude=exclude, include_target=False)
        self.cv = cv
        self.split_kind = split_kind
        self.sort_by = sort_by
        self.ascending = ascending
        self.stratify_by = stratify_by
        self.random_state = random_state
        self.estimator = estimator
        self.target = target
        self.scorers = scorers
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.fit_params = {} if fit_params is None else fit_params
        self.pre_dispatch = pre_dispatch
        self.return_train_score = return_train_score
        self.return_estimator = return_estimator
        self.return_test_indices = return_test_indices
        self.return_predictions = return_predictions
        self.error_score = error_score
        self.proba = proba

    def cross_validate(self, df):
        """
        Calls the _cross_validate method to perform cross-validation.

        :param df: dataframe to perform cross-validation on
        :return: a dictionary containing the summary of the cross-validation
        """
        with Lock.lock():
            result = self._cross_validate(df)
        return result

    def _cross_validate(self, df):
        """
        Performs cross-validation by calling _fit_and_score for each K-fold split. Achieves parallelization by making
        use of joblib.Parallel class.

        :param df: dataframe to perform cross-validation on
        :return: a dictionary containing the summary of the cross-validation
        """
        X, y = df.separate(self.target, to_numpy=True)

        proba = validate_proba(self.proba, df, self.estimator)
        scorers, scorer_names = validate_scorers(self.scorers, proba)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)
        results = parallel(
            delayed(_fit_and_score)(
                clone(self.estimator),
                X,
                y,
                scorers,
                train,
                test,
                self.verbose,
                None,
                self.fit_params,
                return_train_score=self.return_train_score,
                return_times=True,
                return_estimator=self.return_estimator,
                return_test_indices=self.return_test_indices,
                return_predictions=self.return_predictions,
                error_score=self.error_score,
                proba=proba
            )
            for train, test in df.kfold_split(n_splits=self.cv,
                                              kind=self.split_kind,
                                              sort_by=self.sort_by,
                                              ascending=self.ascending,
                                              stratify_by=self.stratify_by,
                                              return_indices=True,
                                              random_state=self.random_state)
        )

        _warn_or_raise_about_fit_failures(results, self.error_score)

        # For callable scoring, the return type is only known after calling. If the
        # return type is a dictionary, the error scores can now be inserted with
        # the correct key.
        if callable(self.scorers):
            _insert_error_scores(results, self.error_score)

        results = _aggregate_score_dicts(results)

        ret = {"fit_time": results["fit_time"], "score_time": results["score_time"], "scorer_names": scorer_names}

        if self.return_estimator:
            ret["estimator"] = results["estimator"]

        if self.return_test_indices:
            ret["test_indices"] = results["test_indices"]

        if self.return_predictions:
            ret["predictions"] = results["predictions"]

        test_scores_dict = _normalize_score_results(results["test_scores"])
        if self.return_train_score:
            train_scores_dict = _normalize_score_results(results["train_scores"])

        for name in test_scores_dict:
            ret["test_%s" % name] = test_scores_dict[name]
            if self.return_train_score:
                key = "train_%s" % name
                ret[key] = train_scores_dict[name]

        return ret

    def __eq__(self, other):
        return self.cv == other.cv \
               and self.split_kind == other.split_kind \
               and self.random_state == other.random_state \
               and self.estimator == other.estimator \
               and self.include == other.include \
               and self.exclude == other.exclude \
               and self.auto == other.auto \
               and self.target == other.target \
               and self.scorers == other.scorers \
               and self.n_jobs == other.n_jobs \
               and self.verbose == other.verbose \
               and self.fit_params == other.fit_params \
               and self.pre_dispatch == other.pre_dispatch \
               and self.return_train_score == other.return_train_score \
               and self.return_estimator == other.return_estimator \
               and self.error_score == other.error_score \
               and self.proba == other.proba


def _fit_and_score(
        estimator,
        X,
        y=None,
        scorers = [],
        train: list = [],
        test: list = [],
        verbose: int = 0,
        params: dict | None = None,
        fit_params: dict | None = None,
        return_train_score: bool = False,
        return_parameters: bool = False,
        return_n_test_samples: bool = False,
        return_times: bool = False,
        return_estimator: bool = False,
        return_test_indices: bool = False,
        return_predictions: bool = False,
        split_progress: list | tuple | None = None,
        candidate_progress: list | tuple | None = None,
        error_score=np.nan,
        proba: bool = False
):
    """
    Fits and scores an estimator for a given split and returns a dict containing the summary.

    :param estimator: estimator object used to fit the data and make predictions
    :param df: dataframe containing the data
    :param scorers: scoring function(s) as callable or list of callables to use for cross-validation. Each callable
    function has to take two vectors as inputs and return a single value
    :param train: indices of the training samples
    :param test: indices of the test samples
    :param verbose: verbosity level
    :param params: parameters to be set on the estimator
    :param fit_params: parameters to pass to the estimator's fit method
    :param return_train_score: if True, train scores will be returned as part the resulting dictionary
    :param return_parameters: if True, estimator parameters will be returned as part the resulting dictionary
    :param return_n_test_samples: if True, number of test samples will be returned as part the resulting dictionary
    :param return_times: if True, train fit and score times will be returned as part the resulting dictionary
    :param return_estimator: if True, the fitted estimator will be returned as part the resulting dictionary
    :param return_test_indices: if True, indices of the test samples will be returned as part the resulting dictionary
    :param return_predictions: if True, test set predictions will be returned as part the resulting dictionary
    :param split_progress: a list or tuple of format (<current_split_id>, <total_num_of_splits>)
    :param candidate_progress: a list or tuple of format (<current_candidate_id>, <total_number_of_candidates>)
    :param error_score: value to assign to the score if an error occurs in estimator fitting
    :param proba: if True, the estimator will be fitted and predictions made using the predict_proba method
    :return: a dictionary containing the summary of the fit and score
    """
    fit_params = {} if fit_params is None else fit_params
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0] + 1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0] + 1}/{candidate_progress[1]}"

    if verbose > 1:
        if params is None:
            params_msg = ""
        else:
            sorted_keys = sorted(params)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={params[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    if params is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in params.items():
            cloned_parameters[k] = clone(v)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train = X[train, :]
    X_test = X[test, :]
    y_train, y_test = None, None
    if y is not None:
        y_train = y[train]
        y_test = y[test]

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            test_scores = error_score
            if return_train_score:
                train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores, y_pred_test = _score(estimator, X_test, y_test, scorers, error_score, proba)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores, y_pred_train = _score(estimator, X_train, y_train, scorers, error_score, proba)

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_test_indices:
        result["test_indices"] = test
    if return_predictions:
        result["predictions"] = y_pred_test
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = len(test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = params
    if return_estimator:
        result["estimator"] = estimator
    return result
