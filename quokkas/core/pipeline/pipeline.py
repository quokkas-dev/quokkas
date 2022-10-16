from __future__ import annotations

import pickle

import numpy as np

from .completion import Completion
from .inception import Inception
from .lock import Lock
from ...transforms.mapper import Mapper
from ...utils.other_utils import validate_scorers, validate_proba
from ...utils.sk_utils import _score


class Pipeline:
    """
    Pipeline of operations on a dataframe
    """

    def __init__(self, transformations: list = None, inception: Inception | None = None,
                 completion: Completion | None = None, model=None, encoded_cols=None):
        """
        :param transformations: list of transformation objects
        :param inception: if provided, will be used as the inception
        of the pipeline
        :param completion: if provided, will be used as the completion
        of the pipeline
        :param model: if provided, will be used as the model in the
        pipeline
        :param encoded_cols: if provided, these columns will be considered
        to be encoded
        """

        self._transformations = transformations
        self._enabled = True
        self._disable_count = 0
        self._inception = inception
        self._completion = completion
        self.model = model
        self._fit_inplace = False
        self._fit_to_numpy = True
        self._squeeze = True

        self._encoded_cols = encoded_cols if encoded_cols is not None else set()

    def __enter__(self):
        """
        Locks this pipeline from any changes via context manager

        :return: locked pipeline
        """
        self._enabled = False
        self._disable_count += 1
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Unlocks this pipeline

        """
        self._disable_count -= 1
        if self._disable_count < 1:
            self._enabled = True
        if exc_type is None:
            return True

    def __repr__(self):
        spaces = 2
        output = 'Pipeline (\n'
        if self._inception is not None:
            output += ' ' * spaces + '(inception) (\n'
            output += ' ' * 2 * spaces + repr(self._inception) + '\n' + spaces * ' ' + ')\n'

        if self._transformations is not None:
            output += ' ' * spaces + '(transformations) (\n'
            for i, transformation in enumerate(self._transformations):
                output += 2 * spaces * ' ' + '(' + str(i) + ') ' + repr(transformation) + '\n'
            output += ' ' * spaces + ')\n'

        if self._completion is not None:
            output += ' ' * spaces + '(completion) (\n'
            output += ' ' * 2 * spaces + repr(self._completion) + '\n' + spaces * ' ' + ')\n'

        if self.model is not None:
            output += ' ' * spaces + '(model) (\n'
            output += ' ' * 2 * spaces + repr(self.model) + '\n' + spaces * ' ' + ')\n'

        output += ')'

        return output

    def equals(self, other):
        return type(self) == type(other) \
               and self._inception == other._inception \
               and self._completion == other._completion \
               and (self._transformations is None and other._transformations is None) or (
                       self._transformations is not None and other._transformations is not None and all(
                   [x.equals(y) for x, y in zip(self._transformations, other._transformations)])) \
               and self.model == other.model

    # incept and complete

    def add_inception(self, func, *args, **kwargs):
        """
        Adds inception to the pipeline

        :param func: function to be used as inception
        :param args: arguments to be provided to the incepting
        function
        :param kwargs: keyword arguments to be provided to the
        incepting function
        """
        if Lock._unlocked:
            self._inception = Inception(func, *args, **kwargs)

    def reincept(self, *args, **kwargs):
        """
        Applies saved inception and creates a new Functional
        :param args: arguments that will be passed to the stored
        incept function
        :param kwargs: keyword arguments that will be passed to
        the stored incept function
        :return: Functional created by the stored incept function
        """
        if self._inception is None:
            raise RuntimeError('Inception is not specified. Please add an inception first')

        df = self._inception(*args, **kwargs)
        if Lock._unlocked and df.pipeline._inception is None:
            df.pipeline._inception = self._inception

        return df

    @classmethod
    def incept(cls, func_name, *args, **kwargs):
        """
        Incepts a dataframe via provided function and its
        arguments

        :param func_name: function to be used (origin)
        :param args: arguments to be provided to the
        incepting function
        :param kwargs: keyword arguments to be provided
        to the incepting function
        :return: created dataframe
        """
        inception = Inception(func_name, *args, **kwargs)
        df = inception()
        if Lock._unlocked:
            df.pipeline._inception = inception
        return df

    def add_completion(self, func, *args, **kwargs):
        """
        Adds a completion to the pipeline

        :param func: function that will be used to complete
        dataframes
        :param args: arguments to be used by completing function
        :param kwargs: keyword arguments to be used by completing function
        """
        if Lock._unlocked and self._enabled:
            self._completion = Completion(func, *args, **kwargs)

    def del_completion(self):
        """
        Deletes stored completion from the pipeline

        """
        self._completion = None

    def complete(self, df, *args, **kwargs):
        """
        Utilizes stored completion to complete (potentially another)
        dataframe

        :param df: dataframe to be completed
        :param args: additional arguments which may be used to replace
        the initially stored arguments
        :param kwargs: additional keyword arguments which may be used to
        replace the initially stored keyword arguments
        :return:
        """
        if self._completion is None:
            raise RuntimeError('Completion is not specified. Please add a completion first')
        return self._completion(df, *args, **kwargs)

    # model fit and predict logic

    def _update_internal_params(self, to_numpy, inplace, squeeze):
        self._fit_to_numpy = self._fit_to_numpy if to_numpy is None else to_numpy
        self._fit_inplace = self._fit_inplace if inplace is None else inplace
        self._squeeze = self._squeeze if squeeze is None else squeeze

    def fit_model(self,
                  df,
                  model,
                  *args,
                  to_numpy: bool | None = None,
                  inplace: bool | None = None,
                  squeeze: bool | None = None,
                  **kwargs):
        """
        Fits the provided model to the provided dataframe. Separates the data
        to target / non-target features, (if requested) transforms them to numpy
        and squeezes, and fits the model

        :param df: dataframe to be fitted
        :param model: model to be fitted
        :param args: additional arguments to be provided to the model.fit function
        :param to_numpy: if True, the data will be transformed to numpy first
        :param inplace: if True, the data will be separated inplace
        :param squeeze: if True, the dimensions of size 1 in the resulting target /
        non-target arrays will be collapsed
        :param kwargs: additional arguments to be provided to the model.fit function
        """


        self._update_internal_params(to_numpy, inplace, squeeze)
        with self:
            X, y = df.separate(to_numpy=self._fit_to_numpy,
                               inplace=self._fit_inplace,
                               ignore_target=False,
                               squeeze=self._squeeze)
            self.model = model
            return self.model.fit(X, *args, **kwargs) if y is None else self.model.fit(X, y, *args, **kwargs)

    def predict(self, df, model=None, to_numpy: bool | None = None, inplace: bool | None = None,
                proba: bool | None = None, squeeze: bool | None = None, **kwargs):
        """
        Predicts the target variable from the dataframe data via saved /
        provided model

        :param df: dataframe to be used for predictions
        :param args: additional arguments to be provided to the model.predict function TODO: delete or implement
        :param model: if provided, this model will be used to make predictions. Otherwise,
        the model saved in the pipeline will be used instead
        :param to_numpy: if True, the data will be converted to numpy
        first. If None, the 'to_numpy' value of fit will be used
        :param inplace: if True, the data will be separated inplace (if
        the separation is necessary). If None, the 'inplace' value of
        fit will be used
        :param proba: if True, will use the predict_proba method of the model. If False,
        will use the predict method. If None, predict_proba will be used if the model
        has that method, otherwise predict
        :param squeeze: if True, the separated features / target will
        be squeezed of dims with length 1. If None, the 'squeeze' value of
        fit will be used
        :param kwargs: additional kw arguments to be provided to predict
        function of the model
        :return: predictions as produced by the model
        """
        if model is not None:
            self.model = model  # TODO: is it not better not to overwrite the saved model here? see evaluate
        if self.model is None:
            raise RuntimeError('Model is not specified. Please add a model first')

        self._update_internal_params(to_numpy, inplace, squeeze)

        proba = validate_proba(proba, df, self.model)

        with self:
            X = df.separate(inplace=self._fit_inplace,
                            to_numpy=self._fit_to_numpy,
                            ignore_target=True,
                            squeeze=self._squeeze)

        return self.model.predict_proba(X, **kwargs) if proba else self.model.predict(X, **kwargs)

    def evaluate(self, df, model=None, scorers=None, error_score: float | int | np.nan | None = np.nan,
                 to_numpy: bool | None = None, inplace: bool | None = None, proba: bool | None = None,
                 squeeze: bool | None = None, return_predictions: bool = False, **kwargs):
        """
        Evaluates the model on the provided dataframe

        :param df: dataframe to be used for evaluation
        :param model: if provided, this model will be used to make predictions. Otherwise,
        the model saved in the pipeline will be used instead
        :param scorers: scoring function(s) as callable or list of callables to use for cross-validation. Each callable
        function has to take two vectors as inputs and return a single value
        :param error_score: value to assign to the score if an error occurs. If 'raise', the error is raised
        :param to_numpy: if True, the data will be converted to numpy
        first. If None, the 'to_numpy' value of fit will be used
        :param inplace: if True, the data will be separated inplace (if
        the separation is necessary). If None, the 'inplace' value of
        fit will be used
        :param proba: if True, will use the predict_proba method of the model. Otherwise
        will use the predict function
        :param squeeze: if True, the separated features / target will
        be squeezed of dims with length 1. If None, the 'squeeze' value of
        fit will be used
        :param return_predictions: if True, will return the predictions as well as the scores
        :param kwargs: additional kw arguments to be provided to predict
        function of the model
        :return: predictions as produced by the model
        """
        model = self.model if model is None else model
        if model is None:
            raise RuntimeError('Model is not specified. Please add a model first')
        proba = validate_proba(proba, df, model)
        scorers, scorer_names = validate_scorers(scorers, proba)
        with self:
            X, y = df.separate(inplace=inplace, to_numpy=to_numpy, ignore_target=False, squeeze=squeeze)
        result = dict()
        result['scores'], pred = _score(model, X, y, scorers, error_score=error_score, predict_proba=proba)
        if return_predictions:
            result['predictions'] = pred
        result["scorer_names"] = scorer_names
        return result

    # save and load logic

    def save(self, path) -> None:
        """
        Saves the pipeline at a specified path

        :param path: str or path-like
        """
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path) -> Pipeline:
        """
        Loads the pipeline from a specified path

        :param path: str or path-like
        :return: loaded pipeline
        """
        with open(path, 'rb') as handle:
            pipeline = pickle.load(handle)
        return pipeline

    # pipeline fit and transform logic

    def fit(self, df):
        """
        Fits the pipeline to provided data

        :param df: dataframe to be transformed
        """
        if self._transformations is None:
            raise ValueError("Pipeline transformations were not initialized")
        df_copy = df.copy(deep=True)
        self.fit_transform(df_copy)

    def fit_transform(self, df, parameters=None, start: int = 0):
        """
        Fits the pipeline to provided data and transforms the underlying data
        accordingly. Also see the transform documentation.

        :param df: dataframe to be transformed
        :param parameters: list of tuples or tuple of objects to be passed
        to streamers
        :param start: (int) index of the first transform
        :return: transformed dataframe

        """

        if self._transformations is None:
            raise ValueError("Pipeline transformations were not initialized")
        for i in range(start, len(self._transformations)):
            if isinstance(self._transformations[i], Mapper):
                df = Pipeline._map_transform(df, self._transformations[i], parameters)
            else:
                if hasattr(self._transformations[i], 'inplace') and self._transformations[i].inplace:
                    self._transformations[i].fit_transform(df)
                else:
                    df = self._transformations[i].fit_transform(df)
        return df

    def transform(self, df, parameters=None, start: int = 0):
        """
        Transforms provided data according to already saved transformations.

        :param df: dataframe to be transformed
        :param parameters: list of tuples or tuple of objects to be passed to streamers
        :param start: (int) index of the first transform
        :return: transformed dataframe
        """

        if self._transformations is None:
            raise ValueError("Pipeline transformations were not initialized")
        for i in range(start, len(self._transformations)):
            if isinstance(self._transformations[i], Mapper):
                df = Pipeline._map_transform(df, self._transformations[i], parameters)
            else:
                if hasattr(self._transformations[i], 'inplace') and self._transformations[i].inplace:
                    self._transformations[i].transform(df)
                else:
                    df = self._transformations[i].transform(df)
        return df

    @staticmethod
    def _map_transform(df, transform: Mapper, parameters):
        """
        Enables passing of new parameters to an existing mapper

        :param df: dataframe to be transformed
        :param transform: mapper to be used
        :param parameters: parameters to be pushed to all mappers
        in the pipeline
        :return: result of the mapper
        """
        if parameters is None:
            return transform.transform(df)
        if isinstance(parameters, list):
            if len(parameters) == 0:
                return transform.transform(df)
            op = parameters.pop()
        else:
            op = parameters
        if isinstance(op, tuple):
            return transform.transform(df, *op)
        else:
            return transform.transform(df, op)

    # transformations logic

    def add(self, transformation) -> Pipeline:
        """
        Adds a new transformation to the pipeline

        :param transformation: initialized transformation to be added
        :return: self
        """
        if Lock._unlocked and self._enabled:
            if self._transformations is None:
                self._transformations = [transformation]
            else:
                self._transformations.append(transformation)
        return self

    def extend(self, transformations: list) -> Pipeline:
        """
        Adds a list of new transformations to the pipeline

        :param transformations: list of initialized transformations to be added
        :return self
        """
        if Lock._unlocked and self._enabled:
            if self._transformations is None:
                self._transformations = [t for t in transformations]
            self._transformations.extend(transformations)
        return self

    def concat(self, pipeline: Pipeline) -> Pipeline:
        """
        Concatenates the current pipeline transformations with the
        transformations of the provided pipeline

        :param pipeline: a pipeline to be added to a given pipeline
        :return self
        """
        return self.extend(pipeline._transformations)

    def disable(self):
        """
        Disables the pipeline. No transformations, completions or
        inceptions will be saved
        """
        self._enabled = False

    def enable(self):
        """
        Enables the pipeline

        """
        self._enabled = True

    def is_empty(self):
        """
        Returns True if there are no transformations in the
        pipeline, otherwise False

        :return: if the transformations of the pipeline are empty
        """
        return self._transformations is None or not self._transformations

    def find_first_different_transformation(self, other: Pipeline) -> int:
        """
        Finds the first transformation in the other pipeline such that
        it does not correspond to the transformation in this pipeline

        :param other: another pipeline, with which the transformations of
        the current pipeline will be matched
        :return: -1 if there are no transformations in the other, the index
        of the first differing transformation in the other otherwise
        """
        if self.is_empty():
            return 0
        self_len = len(self._transformations)
        for i in range(len(other._transformations)):
            if i == self_len or not self._transformations[i].equals(other._transformations[i]):
                return i
        return -1

    # encoded_cols feature
    def append_encoded_cols(self, cols: list | set | tuple):
        """
        Saves encoded columns. Is used for encoders / date encoders,
        so that the already encoded columns can be ignored

        :param cols: the list, set or tuple of the encoded columns
        :return: self
        """
        self._encoded_cols = self._encoded_cols.union(set(cols))
        return self

    def add_encoded_col(self, col):
        """
        Adds one new encoded column to the stored encoded columns

        :param col: column to be added
        """
        self._encoded_cols.add(col)

    def reduce_encoded_cols(self, cols: set):
        """
        Returns columns that are not in saved encoded columns
        from a provided set of columns

        :param cols: set of columns to be reduced
        :return: reduced set of columns
        """
        return cols - self._encoded_cols
