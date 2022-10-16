from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, Union

import numpy as np
from joblib import Parallel, delayed
from numpy.random import RandomState
from pandas._libs import lib  # only used for no_default

from ..pipeline.pipeline import Pipeline
from ...eda.categorical import CategoricalDetector
from ...eda.correlation import CorrelationVisualizer
from ...eda.distributions import DistVisualizer
from ...eda.feature_importance import ContinuousFeatureImportanceEstimator, CategoricalFeatureImportanceEstimator
from ...eda.generic import _BaseExplainer
from ...eda.missing import MissingValuesVisualizer
from ...eda.scatters import ScatterVisualizer
from ...transforms.date_encoder import DateEncoder
from ...transforms.encoders import OrdinalEncoder, OneHotEncoder
from ...transforms.external import External
from ...transforms.generic import _BaseProcessor
from ...transforms.imputers import SimpleImputer, IterativeImputer, KNNImputer
from ...transforms.mapper import Mapper
from ...transforms.normalizers import Normalizer
from ...transforms.operation import Operation
from ...transforms.scalers import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from ...transforms.splitters import ShuffledSplitter, StratifiedSplitter, SortedSplitter, SequentialSplitter, \
    _BaseSplitter
from ...transforms.trimmers import Trimmer, Winsorizer
from ...transforms.validation import CrossValidator
from ...utils.other_utils import sum_not_nones, create_generator, create_split, validate_scorers, validate_proba
from ...utils.pd_utils.pd_typing import Dtype
from ...utils.sk_utils import _aggregate_score_dicts
from ...utils.tuning_utils import clone_model, pretty_results, _fit_and_score_separated, \
    separate_params


class Functional:
    SCALER_MAP = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'maxabs': MaxAbsScaler,
        'robust': RobustScaler
    }

    IMPUTER_MAP = {
        'simple': SimpleImputer,
        'iterative': IterativeImputer,
        'knn': KNNImputer
    }

    ENCODER_MAP = {
        'ordinal': OrdinalEncoder,
        'onehot': OneHotEncoder
    }

    SPLITTER_MAP = {
        'stratified': StratifiedSplitter,
        'sorted': SortedSplitter,
        'sequential': SequentialSplitter,
        'shuffled': ShuffledSplitter
    }

    VALIDATOR_MAP = {
        'cross_validator': CrossValidator
    }

    FEATURE_IMPORTANCE_MAP = {
        'continuous': ContinuousFeatureImportanceEstimator,
        'categorical': CategoricalFeatureImportanceEstimator
    }

    global_inplace = False

    def __init__(self, pipeline, target):
        if pipeline is None:
            object.__setattr__(self, 'pipeline', Pipeline())
        else:
            object.__setattr__(self, 'pipeline', pipeline)
        if target is None:
            object.__setattr__(self, 'target', set())
        else:
            object.__setattr__(self, 'target', target)

    def preserve(self, func_name, *args, **kwargs):
        self.pipeline.add(Operation(func_name, *args, **kwargs))

    def targetize(self, target):
        if target is None:
            object.__setattr__(self, 'target', set())
        elif isinstance(target, (list, set, tuple)):
            object.__setattr__(self, 'target', set(target))
        else:
            object.__setattr__(self, 'target', {target})

    def map(self, func, *args, **kwargs) -> Functional:
        """
        Streams changes of the underlying dataframe, defined in func
        to its pipeline. The pipeline of the original dataframe is
        preserved unless the operations within func happen inplace or
        func destroys the pipeline. The pipeline changes within func
        are ignored. The additional arguments provided to stream can
        be (but don't need to be) changed when called from a pipeline.

        :param func: function to be applied on a dataframe. Expects df as
        the first argument.
        :param args: additional arguments for the function. DataFrame
        should not be included.
        :param kwargs: additional keyword arguments for the function
        :return: transformed DataFrame with an updated pipeline
        """

        streamer = Mapper(func, *args, **kwargs)
        df = streamer.transform(self, *args, **kwargs)
        return df

    def external(self, processor, inplace: bool = False, fit: bool = True) -> Functional:
        wrapper = External(processor, inplace)
        if fit:
            result = wrapper.fit_transform(self)
        else:
            result = wrapper.transform(self)
        return self if inplace else result

    def _update_and_transform_map(self, processor, map, fit, kind, auto, inplace, *args, **kwargs):
        if processor is None:

            if kind not in map:
                raise ValueError(f'unknown strategy {kind} given')

            if inplace == lib.no_default:
                inplace = self.global_inplace

            processor = map[kind](*args, auto=auto, inplace=inplace, **kwargs)
        if fit:
            result = processor.fit_transform(self)
        else:
            result = processor.transform(self)
        return self if processor.inplace else result

    def _map(self, transformer, map, kind, *args, **kwargs):
        if transformer is None:

            if kind not in map:
                raise ValueError(f'unknown strategy {kind} given')

            transformer = map[kind](*args, **kwargs)
        return transformer

    def _update_and_transform(self, processor, klass, fit, auto, inplace, *args, **kwargs):
        if processor is None:

            if inplace == lib.no_default:
                inplace = self.global_inplace

            processor = klass(*args, auto=auto, inplace=inplace, **kwargs)
        if fit:
            result = processor.fit_transform(self)
        else:
            result = processor.transform(self)
        return self if processor.inplace else result

    def scale(self,
              kind: str = 'standard',
              include: set | list | None = None,
              exclude: set | list | None = None,
              include_target: bool = False,
              auto: bool = True,
              inplace: bool | lib.NoDefault = lib.no_default,
              fast_transform: bool = False,
              processor: _BaseProcessor = None,
              fit: bool = True,
              **kwargs) -> Functional:
        """
        Scales the dataframe according to a provided scaler / kind of scaler.
        Supports the following kinds:

        - standard (equivalent to StandardScaler), supports additional
         'with_mean' and 'with_std' arguments

        - robust (equivalent to RobustScaler), supports additional
        'with_centering', 'with_scaling' and 'quantile_range' arguments

        - minmax (equivalent to MinMaxScaler), supports additional
        'feature_range' and 'clip' arguments

        - maxabs (equivalent to MaxAbsScaler)

        :param kind: type of scaler to be used (need to be specified if
        no scaler is provided), default 'standard'
        :param include: if provided, the transform will be applied only to these columns
        :param exclude: if provided, the transform will not be applied to these columns
        :param include_target: if True, the target column will be scaled too
        :param auto: if True, the transform will only be attempted for numeric columns
        :param inplace: if True, the transform will be completed inplace
        :param fast_transform: attempts to transform the data as-is -
        in particular, no auto-detection of columns to be transformed will be
        attempted. 'include' or 'exclude' arguments cannot be provided together
        with fast_transform. If a target is provided and 'include_target' is set to
        False, the data won't be transformed 'as-is'. Default False.
        :param processor: if provided, the transformation will be completed
        with this scaler. Default None.
        :param fit: if the provided scaler should fit the data first
        :param kwargs: additional kw arguments to be provided to the scaler
        :return: transformed dataframe
        """

        return self._update_and_transform_map(processor, Functional.SCALER_MAP, fit, kind, auto, inplace,
                                              include=include, exclude=exclude, include_target=include_target,
                                              fast_transform=fast_transform, **kwargs)

    def impute(self,
               kind: str = 'simple',
               include: set | list | None = None,
               exclude: set | list | None = None,
               include_target: bool = False,
               auto: bool = True,
               inplace: bool | lib.NoDefault = lib.no_default,
               processor: _BaseProcessor = None,
               fit: bool = True,
               missing_values: Union[str, float, int, np.nan, None] = np.nan,
               **kwargs) -> Functional:
        """
        Imputes missing values in the dataframe according to the provided imputer /
        kind of imputer. Supports the following kinds:

        - simple (equivalent to SimpleImputer), supports additional arguments 'strategy', 'fill_value'
        and 'fast_transform'
        - iterative (equivalent to IterativeImputer), supports additional arguments 'sample_posterior', 'max_iter',
        'tol', 'n_nearest_features', 'initial_strategy', 'imputation_order, 'skip_complete, 'min_value',
        'max_value' and 'random_state'
        - knn (equivalent to KNNImputer), supports additional arguments 'n_neighbors', 'weights' and 'metric'

        :param kind: type of imputer to be used (need to be specified if
        no imputer is provided), default 'simple'
        :param include: if provided, the transform will be applied only to these columns
        :param exclude: if provided, the transform will not be applied to these columns
        :param include_target: if True, the target column will be imputed too
        :param auto: if True, the transform will only be attempted for numeric columns
        :param inplace: if True, the transform will be completed inplace
        :param processor: if provided, the transformation will be completed
        with this imputer. Default None.
        :param fit: if the provided imputer should fit the data first
        :param missing_values: the value to be used to fill missing values.
        :param kwargs: additional kw arguments to be provided to the imputer
        :return: transformed dataframe
        """

        return self._update_and_transform_map(processor, Functional.IMPUTER_MAP, fit, kind, auto, inplace,
                                              include=include, exclude=exclude, include_target=include_target,
                                              missing_values=missing_values, **kwargs)

    def encode(self,
               kind: str = 'ordinal',
               include: set | list | None = None,
               exclude: set | list | None = None,
               include_target: bool = False,
               auto: bool = True,
               inplace: bool | lib.NoDefault = lib.no_default,
               processor: _BaseProcessor = None,
               fit: bool = True,
               dtype: Dtype = np.float64,
               categories: str | dict = 'auto',
               handle_unknown: str = 'error',
               **kwargs) -> Functional:
        """
        Encodes the categorical values in the dataset according to the provided encoder
        or the provided type of encoder. Supports the following kinds:

        - ordinal (equivalent to OrdinalEncoder), supports additional unknown_value
        (value for columns that were not encoded before, default None), and
        encoded_missing_value (value for encoding of nans, if None, nans are encoded
        like any other number, default None) arguments

        - onehot (equivalent to OnehotEncoder), supports additional drop
        (if 'first' - drops first encoded column, if 'if_binary' - drops
        first encoded column only if binary, if None - doesn't drop any),
        min_frequency (frequency of value to not be considered an
        infrequent column, can be between 0 and 1 or an integer),
        max_categories (maximum number of allowed categories, all others
        will be considered infrequent), sparse (if True, the encoded
        columns will be pd SparseSeries), infix (string literal to be used
        in the column name) and keep_original (if True, original columns will
        be kept) arguments

        :param kind: type of encoder to be used (need to be specified if
        no imputer is provided), default 'simple' # todo: is that really a correct description?
        :param include: if provided, the transform will be applied only to these columns
        :param exclude: if provided, the transform will not be applied to these columns
        :param include_target: if True, the target column will be encoded too
        :param auto: if True, the transform will only be attempted for columns that can
        be identified as categorical. Columns are identified as categorical if there are
        20 or fewer values and the share of distinct values is less than 10%
        :param inplace: if True, the transform will be completed inplace
        :param processor: if provided, the transformation will be completed
        with this encoder. Default None.
        :param fit: if the provided encoder should fit the data first
        :param dtype: the type to which the columns will be encoded
        :param categories: the values of the categories in each column to be
        encoded as a dictionary
        :param handle_unknown: the strategy to deal with not previously encountered
        values in the encoder. Can be 'error' or 'use_encoded_value'
        for ordinal encoder and 'error', 'ignore' and 'infrequent_if_exist' for
        onehot encoder
        :param kwargs: additional kw arguments to be provided to the encoder
        :return: transformed dataframe
        """

        return self._update_and_transform_map(processor, Functional.ENCODER_MAP, fit, kind, auto, inplace,
                                              include=include, exclude=exclude, include_target=include_target,
                                              dtype=dtype, categories=categories, handle_unknown=handle_unknown,
                                              **kwargs)

    def encode_dates(self,
                     include: set | list | None = None,
                     exclude: set | list | None = None,
                     include_target: bool = False,
                     auto: bool = True,
                     inplace: bool | lib.NoDefault = lib.no_default,
                     encoder: DateEncoder = None,
                     fit: bool = True,
                     ordinal=True,
                     intrayear=False,
                     intraweek=False,
                     keep_original=False,
                     ):
        """
        Encodes the date columns in the dataset. Allows to map the dates
        to a [0, 1] interval, map the days in a year into a [0, 1] interval
        (to account for seasonal effects) and map the day of the week to a
        [0, 6] interval (to account for weekday effects).

        :param include: if provided, the transform will be applied only to these columns
        :param exclude: if provided, the transform will not be applied to these columns
        :param include_target: if True, the target column will be encoded too
        :param auto: if True, the transform will only be attempted for datetime columns.
        :param inplace: if True, the transform will be completed inplace
        :param encoder: if provided, this DateEncoder will be used
        :param fit: if the provided DateEncoder should fit the data first
        :param ordinal: if the dates should be encoded ordinally, i.e. maximal
        date mapped to one, minimal to 0, all others linearly between them
        :param intrayear: if the days of year should be encoded, i.e.
        31.12. 23:59:59 to 1, 01.01. 00:00:00 to 0, all others linearly between them
        :param intraweek: if the days should be added as day of the week (0 - 6)
        :param keep_original: if the original date columns should be saved
        :return: transformed dataframe
        """

        return self._update_and_transform(encoder, DateEncoder, fit, auto, inplace,
                                          include=include, exclude=exclude, include_target=include_target,
                                          ordinal=ordinal, intrayear=intrayear, intraweek=intraweek,
                                          keep_original=keep_original)

    def normalize(self,
                  include: set | list | None = None,
                  exclude: set | list | None = None,
                  include_target: bool = False,
                  auto: bool = True,
                  inplace: bool | lib.NoDefault = lib.no_default,
                  normalizer: Normalizer = None,
                  fit: bool = True,
                  norm='l2'
                  ):
        """
        Normalizes the dataframe values row-wise. Supports l2, l1 and max (l_inf) norms.

        :param include: if provided, the transform will be applied only to these columns
        :param exclude: if provided, the transform will not be applied to these columns
        :param include_target: if True, the target column will be normalized too
        :param auto: if True, the transform will only be attempted for numeric columns
        :param inplace: if True, the transform will be completed inplace
        :param normalizer: if provided, this normalizer will be used to transform data
        :param fit: if True, the provided normalizer will be fitted first
        :param norm: 'l2', 'l1' or 'max'
        :return: transformed dataframe
        """

        return self._update_and_transform(normalizer, Normalizer, fit, auto, inplace,
                                          include=include, include_target=include_target, exclude=exclude, norm=norm)

    def trim(self,
             include: set | list | None = None,
             exclude: set | list | None = None,
             include_target: bool = False,
             auto: bool = True,
             inplace: bool | lib.NoDefault = lib.no_default,
             trimmer: Trimmer = None,
             fit: bool = True,
             limits: tuple | dict = (0.01, 0.01),
             inclusive: tuple = (True, True),
             relative: bool = True):
        """
        Trims the dataframe of "outlier" values.

        :param include: if provided, the dataframe will be trimmed only
        according to these columns
        :param exclude: if provided, the transform will not be trimmed according
        to these columns
        :param include_target: if True, the dataframe will be trimmed according to
        target column as well
        :param auto: if True, the transform will be attempted only for numeric columns
        :param inplace: if True, the transform will be completed inplace
        :param trimmer: if provided, this object will be used to trim the dataframe
        :param fit: if True, the object will be fitted first
        :param limits: if relative is True, the top / bottom percentiles to be excluded
        for each column. If relative is False, all feature values below first /
        above second threshold will be trimmed
        :param inclusive: If True, the values that correspond to the exact limits will
        not be trimmed. If False, they will be trimmed
        :param relative: if True, limits will be interpreted as percentiles. If False,
        the limits will be interpreted as absolute values
        :return: transformed dataframe
        """
        return self._update_and_transform(trimmer, Trimmer, fit, auto, inplace, limits=limits,
                                          include=include, exclude=exclude, include_target=include_target,
                                          inclusive=inclusive, relative=relative)

    def winsorize(self,
                  include: set | list | None = None,
                  exclude: set | list | None = None,
                  include_target: bool = False,
                  auto: bool = True,
                  inplace: bool | lib.NoDefault = lib.no_default,
                  winsorizer: Winsorizer = None,
                  fit: bool = True,
                  limits: tuple | dict = (0.01, 0.01),
                  inclusive: tuple = (True, True),
                  relative: bool = True):
        """
        Winsorizes the dataframe of "outlier" values.

        :param include: if provided, only these columns will be winsorized
        :param exclude: if provided, these columns will not be winsorized
        :param include_target: if True, the target column will be winsorized as well
        :param auto: if True, the dataframe will be winsorized only according to
        numeric columns
        :param inplace: if True, the transform will be completed inplace
        :param winsorizer: if provided, this object will be used to winsorize the
        dataframe
        :param fit: if True, the object will be fitted first
        :param limits: if relative is True, the top / bottom percentiles winsorized
        for each column. If relative is False, all feature values below first /
        above second threshold will be winsorized
        :param inclusive: If True, the values that correspond to the exact limits will
        not be winsorized. If False, they will be winsorized as well
        :param relative: if True, limits will be interpreted as percentiles. If False,
        the limits will be interpreted as absolute values
        :return: transformed dataframe
        """
        return self._update_and_transform(winsorizer, Winsorizer, fit, auto, inplace, limits=limits,
                                          include=include, exclude=exclude, include_target=include_target,
                                          inclusive=inclusive, relative=relative)

    def feature_importance(self,
                           kind: str = 'default',
                           include: set | list | None = None,
                           exclude: set | list | None = None,
                           auto: bool = True,
                           target=None,
                           **kwargs):
        """
        Calculates feature importance as variance of feature means
        among buckets of values less expected variance for independent
        feature values. Buckets are formed as:

        - if kind is 'categorical' - feature values for distinct target values
        - if kind is 'continuous' - feature values for n quantiles of target
        values. Supports 'buckets' additional argument, which represents the
        number of quantiles to be considered
        - if kind is 'default' - the type of target value will be inferred
        based on the number of distinct target values

        :param kind: 'default', 'categorical', or 'continuous'
        :param include: if provided, only those features will be analysed
        :param exclude: if provided, these features won't be analysed
        :param auto: if True, only numeric columns will be considered
        :param target: if provided, this column will be used as target.
        Otherwise, the target of the dataframe will be used
        :param kwargs: additional keyword-arguments to be provided to
        the feature importance estimator
        :return: FeatureImportanceEstimator, can be transformed into
        a dict / list
        """

        if kind == 'default':
            border = CategoricalDetector.get_border(CategoricalDetector.DEFAULT_CAT_NUMBER,
                                                    CategoricalDetector.DEFAULT_CAT_SHARE, self.shape[0])
            if border > self[_BaseExplainer.determine_unique_target(self, target)].nunique(dropna=False):
                kind = 'categorical'
            else:
                kind = 'continuous'

        if kind not in self.FEATURE_IMPORTANCE_MAP:
            raise ValueError(f'Unknown feature importance estimator kind {kind}')

        feature_importance_estimator = self.FEATURE_IMPORTANCE_MAP[kind](include=include,
                                                                         exclude=exclude,
                                                                         auto=auto,
                                                                         **kwargs)

        feature_importance_estimator.fit(self, target)
        return feature_importance_estimator

    def plot_feature_means(self,
                           kind: str = 'default',
                           include: set | list | None = None,
                           exclude: set | list | None = None,
                           auto: bool = True,
                           target=None,
                           figsize: tuple = (12, 5),
                           style: str = 'seaborn',
                           **kwargs):
        """
        Plots means of the features split by buckets.
        - if kind is 'categorical', buckets are equivalent to distinct
        target values
        - if kind is 'continuous', buckets are equivalent to n quantiles
        of target values. Supports 'buckets' additional argument, which
        represents the number of target quantiles to be considered
        - if kind is 'default' - the type of target value will be inferred
        based on the number of distinct target values

        :param kind: 'default', 'categorical', or 'continuous'
        :param include: if provided, only those features will be plotted
        :param exclude: if provided, these features won't be plotted
        :param auto: if True, only numeric columns will be considered
        :param target: if provided, this column will be used as target.
        Otherwise, the target of the dataframe will be used
        :param figsize: the size of the figure that it will be plotted on
        :param style: one of the available plt styles
        :param kwargs: additional kw arguments to be provided to the chart
        """

        if kind == 'default':
            border = CategoricalDetector.get_border(CategoricalDetector.DEFAULT_CAT_NUMBER,
                                                    CategoricalDetector.DEFAULT_CAT_SHARE, self.shape[0])
            if border > self[_BaseExplainer.determine_unique_target(self, target)].nunique(dropna=False):
                kind = 'categorical'
            else:
                kind = 'continuous'
        elif kind not in self.FEATURE_IMPORTANCE_MAP:
            raise ValueError(f'Unknown feature importance estimator kind {kind}')

        feature_importance_estimator = self.FEATURE_IMPORTANCE_MAP[kind](include=include, exclude=exclude,
                                                                         auto=auto, **kwargs)
        feature_importance_estimator.fit(self, target)
        feature_importance_estimator.visualize(figsize=figsize, style=style)

    def plot_missing_values(self,
                            include: set | list | None = None,
                            exclude: set | list | None = None,
                            auto: bool = True,
                            reverse: bool = False,
                            figsize: tuple = (12, 5),
                            style: str = 'seaborn',
                            **kwargs):
        """
        Plots missing values share per each feature. If reverse,
        plots the share of non-missing values instead.

        :param include: if provided, only those features will be plotted
        :param exclude: if provided, these features won't be plotted
        :param auto: if True, only numeric columns will be considered
        :param reverse: if True, the share non-missing values will be plotted
        :param figsize: the size of the figure that it will be plotted on
        :param style: one of the available plt styles
        :param kwargs: additional kw arguments to be provided to the chart
        """

        missing_values_visualizer = MissingValuesVisualizer(include=include, exclude=exclude, auto=auto)
        missing_values_visualizer.fit(self)
        missing_values_visualizer.visualize(figsize=figsize, reverse=reverse, style=style, **kwargs)

    def plot_correlation(self, include: set | list | None = None,
                         exclude: set | list | None = None,
                         auto: bool = True,
                         include_target: bool = True,
                         absolute: bool = False,
                         figsize: tuple = (11, 10),
                         style: str = 'seaborn',
                         annot: bool = False,
                         cbar: bool = True,
                         cmap='coolwarm',
                         **kwargs):
        """
        Plots correlation coefficients of the features.

        :param include: if provided, only those features will be plotted
        :param exclude: if provided, these features won't be plotted
        :param auto: if True, only numeric columns will be considered
        :param include_target: if True, the correlation with target will
        be plotted as well
        :param absolute: if True, the absolute correlation coefficients
        will be plotted
        :param figsize: the size of the figure that it will be plotted on
        :param style: one of the available plt styles
        :param annot: if True, the annotations will be provided
        :param cbar: if True, the cbar will be shown on the heatmap
        :param cmap: cmap to be passed to the heatmap
        :param kwargs: additional arguments to be passed to the heatmap
        """

        crv = CorrelationVisualizer(include=include, exclude=exclude, include_target=include_target, auto=auto)
        crv.fit(self)
        crv.visualize(figsize=figsize, style=style, absolute=absolute, annot=annot, cbar=cbar, cmap=cmap, **kwargs)

    def plot_density(self,
                     include: set | list | None = None,
                     exclude: set | list | None = None,
                     auto: bool = True,
                     target=None,
                     n_line_items: int = 3,
                     legend: bool = True,
                     figsize: tuple | None = None,
                     style: str = 'seaborn',
                     cat_detection_strategy: str = 'count&type',
                     cat_number: int = 20,
                     cat_share: float = 0.1,
                     **kwargs):
        """
        Plot estimated density distribution of the non-categorical features
        for distinct target values. Should be only applied for categorical
        targets.

        :param include: if provided, only those features will be plotted
        :param exclude: if provided, these features won't be plotted
        :param auto: if True, only numeric columns will be considered
        :param target: if provided, this column will be used as target.
        Otherwise, the target of the dataframe will be used
        :param n_line_items: number of plots in one line
        :param legend: if the legend should be included
        :param figsize: the size of the figure that it will be plotted on
        :param style: one of the available plt styles
        :param cat_detection_strategy: 'count', 'type' or 'count&type'. If
        'count', the categorical features will be detected based on the
        number of distinct values (with border being min(cat_share * num_rows,
        cat_number)), if 'type' - based on the type of the column, if
        'count&type' - based on both
        :param cat_number: if 'count' or 'count&type', the features with less
        than this number of distinct values will be considered categorical (if
        cat_share * num_rows is larger than cat_number)
        :param cat_share: if 'count' or 'count&type', the features with less
        than cat_share * num_rows of distinct values will be considered categorical
        (if cat_number is larger than cat_share * num_rows)
        :param kwargs: additional kw arguments to be passed to the plot
        """
        density_visualizer = DistVisualizer(include=include, exclude=exclude, auto=auto,
                                            cat_detection_strategy=cat_detection_strategy,
                                            cat_number=cat_number, cat_share=cat_share)
        density_visualizer.fit(self, target=target)
        density_visualizer.visualize(figsize=figsize, n_line_items=n_line_items, legend=legend, style=style, **kwargs)

    def plot_scatter(self,
                     include: set | list | None = None,
                     exclude: set | list | None = None,
                     auto: bool = True,
                     target=None,
                     n_line_items: int = 3,
                     legend: bool = False,
                     figsize: tuple = None,
                     style: str = 'seaborn',
                     **kwargs):
        """
        Creates scatter plots for each feature vs target

        :param include: if provided, only those features will be plotted
        :param exclude: if provided, these features won't be plotted
        :param auto: if True, only numeric columns will be considered
        :param target: if provided, this column will be used as target.
        Otherwise, the target of the dataframe will be used
        :param n_line_items: number of plots in one line
        :param legend: if the legend should be included
        :param figsize: the size of the figure that it will be plotted on
        :param style: one of the available plt styles
        :param kwargs: additional kw arguments to be passed to the plot
        """

        density_visualizer = ScatterVisualizer(include=include, exclude=exclude, auto=auto)
        density_visualizer.fit(self, target=target)
        density_visualizer.visualize(figsize=figsize, n_line_items=n_line_items, legend=legend, style=style, **kwargs)

    def suggest_categorical(self,
                            strategy: str = 'count&type',
                            include: set | list | None = None,
                            exclude: set | list | None = None,
                            cat_number: int = 20,
                            cat_share: float = 0.1):
        """
        Detects categorical columns.

        :param strategy: 'count', 'type' or 'count&type'. If
        'count', the categorical features will be detected based on the
        number of distinct values (with border being min(cat_share * num_rows,
        cat_number)), if 'type' - based on the type of the column, if
        'count&type' - based on both
        :param include: if provided, only those features will be analysed
        :param exclude: if provided, these features won't be analysed
        :param cat_number: if 'count' or 'count&type', the features with less
        than this number of distinct values will be considered categorical (if
        cat_share * num_rows is larger than cat_number)
        :param cat_share: if 'count' or 'count&type', the features with less
        than cat_share * num_rows of distinct values will be considered categorical
        (if cat_number is larger than cat_share * num_rows)
        :return: list of detected categorical features
        """
        categorical_detector = CategoricalDetector(include=include, exclude=exclude, strategy=strategy,
                                                   cat_number=cat_number, cat_share=cat_share)
        categorical_detector.fit(self)
        return categorical_detector.to_list()

    def split(self,
              kind: str = 'shuffled',
              inplace: bool = False,
              splitter: _BaseSplitter | None = None,
              sizes: Iterable[int] | Iterable[float] | None = None,
              n_splits: int | None = None,
              separate: bool = False,
              to_numpy: bool = False,
              return_indices: bool = False,
              ordinal_indices: bool = True,
              random_state: int | RandomState | None = None,
              **kwargs):
        """
        Splits the dataframe according to a kind or according to provided splitter.
        Supports the following kinds:

        - shuffled (equivalent to ShuffledSplitter),

        - sequential (equivalent to SequentialSplitter),

        - sorted (equivalent to SortedSplitter), supports arguments 'sort_by' (column to be used for sorting;
        if provided, splits will not be generated randomly, instead they will be generated according to the order of
        the column) and 'ascending' (when sort_by is provided, is used to reverse the sorting order),

        - stratified (equivalent to StratifiedSplitter), supports argument 'stratify_by' (if provided, the dataframes
        will be stratified according to the values of this column, and of the target column otherwise).

        :param kind: type of scaler to be used (need to be specified if no splitter is provided)
        :param inplace: if True, the transform will be completed inplace
        :param splitter: if provided, this splitter will be used to split the dataframe
        :param sizes: sizes of the splits to be generated. If not provided, the dataframe will be split in
        'n_splits' approximately equal parts. The size of the last part can be inferred if not provided.
        :param n_splits: number of dataframes to be generated, can be inferred from sizes if not provided
        :param return_indices: if True, the indices of the dataframes will be returned instead of dataframes
        :param ordinal_indices: If True and 'return_indices' is True, returns ordinal indices (as those used in iloc),
        otherwise returns indices that correspond to the actual indices of the dataframe (as those used in loc)
        :param separate: if True and 'return_indices' is False, the returned dataframe will be split into two
        containing features and target respectively
        :param to_numpy: if True, 'return_indices' is False and 'separate' is True, the returned dataframes will be
        converted to a numpy array
        :param random_state: random state used for random splitting (used when kind is 'shuffled' or 'stratified')
        :param kwargs: kw arguments to be provided to the splitter
        :return: transformed dataframe
        """

        return self._map(splitter, Functional.SPLITTER_MAP, kind, inplace=inplace, random_state=random_state,
                         return_indices=return_indices, n_splits=n_splits, separate=separate, to_numpy=to_numpy,
                         sizes=sizes, ordinal_indices=ordinal_indices, **kwargs).split(self)

    def train_test_split(self,
                         kind: str = 'shuffled',
                         inplace: bool = False,
                         splitter: _BaseSplitter | None = None,
                         sizes: Iterable[int] | Iterable[float] | None = None,
                         train_size: float | int | None = None,
                         test_size: float | int | None = None,
                         separate: bool = False,
                         to_numpy: bool = False,
                         return_indices: bool = False,
                         ordinal_indices: bool = True,
                         random_state: int | RandomState | None = None,
                         **kwargs
                         ):
        """
        Splits the dataframe into a training and test set according to provided splitter / splitter kind.
        Supports the following kinds:

        - shuffled (equivalent to ShuffledSplitter),

        - sequential (equivalent to SequentialSplitter),

        - sorted (equivalent to SortedSplitter), supports arguments 'sort_by' (column to be used for sorting;
        if provided, splits will not be generated randomly, instead they will be generated according to the order of
        the column) and 'ascending' (when sort_by is provided, is used to reverse the sorting order),

        - stratified (equivalent to StratifiedSplitter), supports argument 'stratify_by' (if provided, the dataframes
        will be stratified according to the values of this column, and of the target column otherwise).

        :param kind: type of scaler to be used (need to be specified if no splitter is provided)
        :param inplace: if True, the transform will be completed inplace
        :param splitter: if provided, this splitter will be used to split the dataframe
        :param sizes: sizes of the splits to be generated. Alternatively, 'train_size' and 'test_size' or just one of
        the arguments can be provided, as the other will be inferred. If none of the arguments are provided, the
        dataframe will be split according to sizes (0.8, 0.2)
        :param train_size: size of the training set as a fraction of the total size or as an absolute number of samples
        :param test_size: size of the test set as a fraction of the total size or as an absolute number of samples
        :param return_indices: if True, the indices of the dataframes will be returned instead of dataframes
        :param ordinal_indices: If True and 'return_indices' is True, returns ordinal indices (as those used in iloc),
        otherwise returns indices that correspond to the actual indices of the dataframe (as those used in loc)
        :param separate: if True and 'return_indices' is False, the returned dataframe will be split into two
        containing features and target respectively
        :param to_numpy: if True, 'return_indices' is False and 'separate' is True, the returned dataframes will be
        converted to a numpy array
        :param random_state: random state used for random splitting (used when kind is 'shuffled' or 'stratified')
        :param kwargs: kw arguments to be provided to the splitter
        :return: transformed dataframe
        """
        if sizes is None:
            n_samples = self.shape[0]
            if any(x < 0 for x in [train_size, test_size] if x is not None):
                raise ValueError("train_size and test_size cannot be negative.")
            strategy = "float" if any(isinstance(x, float) and x not in [0.0, 1.0] for x in
                                      [train_size, test_size]) or (test_size is None and train_size is None) else "int"
            total = 1 if strategy == "float" else n_samples
            if sum_not_nones([train_size, test_size]) > total:
                raise ValueError(f"Sum of train_size and test_size has to be lower or equal to {total}.")
            if train_size is None and test_size is None:
                train_size = 0.8
                test_size = 0.2
            elif train_size is None:
                train_size = total - test_size
            elif test_size is None:
                test_size = total - train_size
            sizes = [train_size, test_size] if strategy == "int" else \
                [round(train_size * n_samples), n_samples - round(train_size * n_samples)]

        return self._map(splitter, Functional.SPLITTER_MAP, kind, sizes=sizes, inplace=inplace, separate=separate,
                         return_indices=return_indices, random_state=random_state, to_numpy=to_numpy,
                         ordinal_indices=ordinal_indices, **kwargs).split(self)

    def train_val_test_split(self,
                             kind: str = 'shuffled',
                             inplace: bool = False,
                             splitter: _BaseSplitter | None = None,
                             sizes: Iterable[int] | Iterable[float] | None = None,
                             train_size: float | int | None = None,
                             val_size: float | int | None = None,
                             test_size: float | int | None = None,
                             separate: bool = False,
                             to_numpy: bool = False,
                             return_indices: bool = False,
                             ordinal_indices: bool = True,
                             random_state: int | RandomState | None = None,
                             **kwargs
                             ):
        """
        Splits the dataframe into a training, validation and test set according to provided splitter / splitter kind.
        Supports the following kinds:

        - shuffled (equivalent to ShuffledSplitter),

        - sequential (equivalent to SequentialSplitter),

        - sorted (equivalent to SortedSplitter), supports arguments 'sort_by' (column to be used for sorting;
        if provided, splits will not be generated randomly, instead they will be generated according to the order of
        the column) and 'ascending' (when sort_by is provided, is used to reverse the sorting order),

        - stratified (equivalent to StratifiedSplitter), supports argument 'stratify_by' (if provided, the dataframes
        will be stratified according to the values of this column, and of the target column otherwise).

        :param kind: type of scaler to be used (need to be specified if no splitter is provided)
        :param inplace: if True, the transform will be completed inplace
        :param splitter: if provided, this splitter will be used to split the dataframe
        :param sizes: sizes of the splits to be generated. Alternatively, train_size, val_size and test_size or
        two of the arguments can be provided, as the rest will be inferred. If none of the arguments are provided,
        the dataframe will be split according to sizes (0.8, 0.1, 0.1)
        :param train_size: size of the training set as a fraction of the total size or as an absolute number of samples
        :param val_size: size of the validation set as a fraction of the total size or as an absolute number of samples
        :param test_size: size of the test set as a fraction of the total size or as an absolute number of samples
        :param return_indices: if True, the indices of the dataframes will be returned instead of dataframes
        :param ordinal_indices: If True and 'return_indices' is True, returns ordinal indices (as those used in iloc),
        otherwise returns indices that correspond to the actual indices of the dataframe (as those used in loc)
        :param separate: if True and 'return_indices' is False, the returned dataframe will be split into two
        containing features and target respectively
        :param to_numpy: if True, 'return_indices' is False and 'separate' is True, the returned dataframes will be
        converted to a numpy array
        :param random_state: random state used for random splitting (used when kind is 'shuffled' or 'stratified')
        :param kwargs: kw arguments to be provided to the splitter
        :return: transformed dataframe
        """
        if sizes is None:
            n_samples = self.shape[0]
            if any(x < 0 for x in [train_size, val_size, test_size] if x is not None):
                raise ValueError("train_size, val_size and test_size cannot be negative.")
            strategy = "float" if any(isinstance(x, float) and x not in [0.0, 1.0] for x in
                                      [train_size, val_size, test_size]) or \
                                  all(size is None for size in [train_size, val_size, test_size]) else "int"
            total = 1 if strategy == "float" else n_samples
            if sum_not_nones([train_size, val_size, test_size]) > total:
                raise ValueError(f"Sum of train_size, val_size and test_size has to be lower or equal to {total}.")
            if train_size is None and val_size is None and test_size is None:
                train_size = 0.8
                val_size = 0.1
                test_size = 0.1
            if (train_size is None and val_size is None) or \
                    (train_size is None and test_size is None) or \
                    (val_size is None and test_size is None):
                raise ValueError("Two or more of train_size, val_size and test_size cannot be None.")
            elif train_size is None:
                train_size = total - val_size - test_size
            elif val_size is None:
                val_size = total - train_size - test_size
            elif test_size is None:
                test_size = total - train_size - val_size
            sizes = [train_size, val_size, test_size] if strategy == "int" else \
                [round(train_size * n_samples), round(val_size * n_samples),
                 n_samples - round(train_size * n_samples) - round(val_size * n_samples)]
        return self._map(splitter, Functional.SPLITTER_MAP, kind,
                         sizes=sizes, inplace=inplace,
                         random_state=random_state, return_indices=return_indices,
                         separate=separate, to_numpy=to_numpy,
                         ordinal_indices=ordinal_indices, **kwargs).split(self)

    def kfold_split(self,
                    kind: str = 'shuffled',
                    inplace: bool = False,
                    splitter: _BaseSplitter | None = None,
                    sizes: Iterable[int] | Iterable[float] | None = None,
                    n_splits: int | None = None,
                    as_list: bool = False,
                    separate: bool = False,
                    to_numpy: bool = False,
                    return_indices: bool = False,
                    ordinal_indices: bool = True,
                    random_state: int | RandomState | None = None,
                    **kwargs):
        """
        Performs a k-fold split on the dataframe according to provided splitter / splitter kind.
        Supports the following kinds:

        - shuffled (equivalent to ShuffledSplitter),

        - sequential (equivalent to SequentialSplitter),

        - sorted (equivalent to SortedSplitter), supports arguments 'sort_by' (column to be used for sorting;
        if provided, splits will not be generated randomly, instead they will be generated according to the order of
        the column) and 'ascending' (when sort_by is provided, is used to reverse the sorting order),

        - stratified (equivalent to StratifiedSplitter), supports argument 'stratify_by' (if provided, the dataframes
        will be stratified according to the values of this column, and of the target column otherwise).

        :param kind: type of scaler to be used (need to be specified if no splitter is provided)
        :param inplace: if True, the transform will be completed inplace
        :param splitter: if provided, this splitter will be used to split the dataframe
        :param sizes: sizes of the splits to be generated. If not provided, the dataframe will be split in
        'n_splits' approximately equal parts. The size of the last part can be inferred if not provided.
        :param n_splits: number of dataframes to be generated, can be inferred from sizes if not provided
        :param as_list: if True, returns a list of dataframes/arrays, otherwise returns a generator
        :param return_indices: if True, the indices of the dataframes will be returned instead of dataframes
        :param ordinal_indices: If True and 'return_indices' is True, returns ordinal indices (as those used in iloc),
        otherwise returns indices that correspond to the actual indices of the dataframe (as those used in loc)
        :param separate: if True and 'return_indices' is False, the returned dataframe will be split into two
        containing features and target respectively
        :param to_numpy: if True, 'return_indices' is False and 'separate' is True, the returned dataframes will be
        converted to a numpy array
        :param random_state: random state used for random splitting (used when kind is 'shuffled' or 'stratified')
        :param kwargs: kw arguments to be provided to the splitter
        :return: transformed dataframe
        """
        splitter = self._map(splitter, Functional.SPLITTER_MAP, kind, n_splits=n_splits, inplace=inplace,
                             random_state=random_state, sizes=sizes, return_indices=True, **kwargs)
        split = splitter.split(self)
        if return_indices:
            if not ordinal_indices:
                split = [self.index.to_numpy()[x] for x in split]
            if as_list:
                return [create_split(split, i) for i in range(n_splits)]
            return create_generator(split=split, n_splits=n_splits)
        elif separate:
            if as_list:
                indices = [create_split(split, i) for i in range(n_splits)]
                return [self.iloc[x, :].separate(to_numpy=to_numpy) for i in range(len(indices)) for x in indices[i]]
            return create_generator(split=split,
                                    n_splits=n_splits,
                                    return_indices=False,
                                    separate=True,
                                    to_numpy=to_numpy,
                                    df=self)
        else:
            if as_list:
                indices = [create_split(split, i) for i in range(n_splits)]
                return [self.iloc[x, :] for i in range(len(indices)) for x in indices[i]]
            return create_generator(split=split,
                                    n_splits=n_splits,
                                    return_indices=False,
                                    separate=False,
                                    to_numpy=False,
                                    df=self)

    def cross_validate(self,
                       validator: CrossValidator | None = None,
                       kind: str = 'cross_validator',
                       cv: int = 5,
                       split_kind: str = 'shuffled',
                       estimator=None,
                       include: set | list | None = None,
                       exclude: set | list | None = None,
                       auto: bool = True,
                       target: str | None = None,
                       scorers=None,
                       error_score=np.nan,
                       n_jobs: int | None = None,
                       verbose: int = 0,
                       fit_params: dict | None = None,
                       proba: bool = False,
                       pre_dispatch: int | str = "2*n_jobs",
                       return_train_score: bool = False,
                       return_estimator: bool = False,
                       return_test_indices: bool = False,
                       return_predictions: bool = False,
                       random_state: int | RandomState | None = None,
                       **kwargs):
        """
        Performs cross-validation on the provided estimator using metrics provided in the 'scoring' parameter.

        :param validator: if provided, this validator will be used to perform cross-validation
        :param kind: the kind of cross-validator to be used. For now the only possibility is 'cross_validator'
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
        return self._map(validator, Functional.VALIDATOR_MAP, kind, cv=cv, split_kind=split_kind, estimator=estimator,
                         include=include, exclude=exclude, auto=auto, target=target, scorers=scorers,
                         random_state=random_state, n_jobs=n_jobs, verbose=verbose, proba=proba,
                         fit_params=fit_params, pre_dispatch=pre_dispatch, return_train_score=return_train_score,
                         return_estimator=return_estimator, return_test_indices=return_test_indices,
                         return_predictions=return_predictions, error_score=error_score, **kwargs).cross_validate(self)

    def search(self,
               params,
               estimator,
               kind='randomized',
               n_iter=10,
               df_test=None,
               split_kind='shuffled',
               sort_by: str | None = None,
               ascending: bool = True,
               stratify_by: str | None = None,
               train_size: float | int | None = None,
               test_size: float | int | None = None,
               to_numpy: bool = True,
               scorers=None,
               error_score=np.nan,
               fit_params: dict | None = None,
               proba: bool = None,
               pre_dispatch='2*n_jobs',
               n_jobs=None,
               random_state=None,
               return_train_score=False,
               **kwargs
               ):
        """
        Performs hyperparameter search on the provided estimator using metrics provided in the 'scorers' parameter.
        :param params: dictionary with parameters to be searched
        :param estimator: estimator to use for hyperparameter search
        :param kind: the kind of hyperparameter search to be used. Possible values: 'randomized' or 'grid'
        :param n_iter: number of iterations for hyperparameter search
        :param df_test: test dataframe to use for scoring. If None, the dataframe will be split into train and test
            sets using parameters 'train_size', 'test_size', 'split_kind', 'to_numpy' and 'random_state'
        :param split_kind: kind of split to use for cross-validation. Possible values:
            - 'shuffled' (equivalent to ShuffledSplitter),
            - 'sequential' (equivalent to SequentialSplitter),
            - 'sorted' (equivalent to SortedSplitter); supports arguments 'sort_by' (column to be used for sorting;
            if provided, splits will not be generated randomly, instead they will be generated according to the order of
            the column) and 'ascending' (when sort_by is provided, is used to reverse the sorting order),
            - 'stratified' (equivalent to StratifiedSplitter); supports argument 'stratify_by' (if provided, the
            dataframes will be stratified according to the values of this column, and of the target column otherwise)
        :param sort_by: column to be used for sorting when 'split_kind' is 'sorted'; if provided, splits will not be
            generated randomly, instead they will be generated according to the order of the column
        :param ascending: when sort_by is provided, is used to reverse the sorting order
        :param stratify_by: is used when 'split_kind' is 'stratified'; if provided, the dataframes will be stratified
            according to the values of this column, and of the target column otherwise
        :param train_size: size of the train set. If None, the default value of 0.8 will be used
        :param test_size: size of the test set. If None, the default value of 0.2 will be used
        :param to_numpy: if True, the dataframe will be converted to numpy arrays before fitting the estimator
        :param scorers: scoring function(s) as callable or list of callables to use for hyperparameter search. Each
            callable function has to take two vectors as inputs and return a single value
        :param error_score: value to assign to the score if an error occurs. If 'raise', the error is raised
        :param fit_params: parameters to pass to the fit method of the estimator
        :param proba: if True, predict_proba method will be used for prediction instead of predict
        :param pre_dispatch: number of jobs to dispatch to workers. Can be used to control speed and memory
        :param n_jobs: number of jobs to run in parallel. If -1, all CPUs are used. If 1, no parallelization is used
        :param random_state: random state to use for shuffling and stratification if the dataframe is split
        :param return_train_score: if True, the training scores will be added to the returned dictionary
        :param kwargs: additional parameters to pass to the hyperparameter search method
        :return: a DataFrame containing the results of the hyperparameter search
        """
        proba = validate_proba(proba, self, estimator)
        init_params, fit_params = separate_params(kind, params, fit_params, n_iter, random_state)
        scorers, scorer_names = validate_scorers(scorers, proba)

        if df_test is None:
            (X_train, y_train), (X_test, y_test) = self.train_test_split(split_kind,
                                                                         sort_by=sort_by,
                                                                         ascending=ascending,
                                                                         stratify_by=stratify_by,
                                                                         train_size=train_size,
                                                                         test_size=test_size,
                                                                         separate=True,
                                                                         to_numpy=to_numpy,
                                                                         random_state=random_state)
        else:
            X_train, y_train = self.separate(to_numpy=to_numpy)
            X_test, y_test = df_test.separate(to_numpy=to_numpy)

        parallel = Parallel(n_jobs=n_jobs, verbose=False, pre_dispatch=pre_dispatch)
        results = parallel(
            delayed(_fit_and_score_separated)(
                clone_model(estimator, init_param),
                X_train,
                y_train,
                X_test,
                y_test,
                scorers,
                {**kwargs, **fit_param},
                return_train_score=return_train_score,
                error_score=error_score,
                proba=proba)
            for init_param, fit_param in zip(init_params, fit_params)
        )

        results = _aggregate_score_dicts(results)

        return pretty_results(list([{**init_tmp, **fit_tmp} for init_tmp, fit_tmp in zip(init_params, fit_params)]),
                              scorer_names, **results)

    def search_cv(self,
                  params,
                  estimator=None,
                  kind='randomized',
                  n_iter=10,
                  cv: int = 5,
                  split_kind: str = 'shuffled',
                  sort_by: str | None = None,
                  ascending: bool = True,
                  stratify_by: str | None = None,
                  scorers=None,
                  error_score: float | int | np.nan | None = np.nan,
                  fit_params: dict | None = None,
                  proba: bool = None,
                  pre_dispatch: int | str = "2*n_jobs",
                  n_jobs: int = None,
                  random_state: int | RandomState | None = None,
                  return_train_score=False,
                  **kwargs
                  ):
        """
        Performs hyperparameter search with cross-validation on the provided estimator using metrics provided in the
        'scorers' parameter.
        :param params: dictionary with parameters to be searched
        :param estimator: estimator to use for hyperparameter search
        :param kind: the kind of hyperparameter search to be used. Possible values: 'randomized' or 'grid'
        :param n_iter: number of iterations for hyperparameter search
        :param cv: number of folds for cross-validation
        :param split_kind: kind of split to use for cross-validation. Possible values:
            - 'shuffled' (equivalent to ShuffledSplitter),
            - 'sequential' (equivalent to SequentialSplitter),
            - 'sorted' (equivalent to SortedSplitter); supports arguments 'sort_by' (column to be used for sorting;
            if provided, splits will not be generated randomly, instead they will be generated according to the order of
            the column) and 'ascending' (when sort_by is provided, is used to reverse the sorting order),
            - 'stratified' (equivalent to StratifiedSplitter); supports argument 'stratify_by' (if provided, the
            dataframes will be stratified according to the values of this column, and of the target column otherwise)
        :param sort_by: column to be used for sorting when 'split_kind' is 'sorted'; if provided, splits will not be
            generated randomly, instead they will be generated according to the order of the column
        :param ascending: when sort_by is provided, is used to reverse the sorting order
        :param stratify_by: is used when 'split_kind' is 'stratified'; if provided, the dataframes will be stratified
            according to the values of this column, and of the target column otherwise
        :param scorers: scoring function(s) as callable or list of callables to use for hyperparameter search. Each
            callable function has to take two vectors as inputs and return a single value
        :param error_score: value to assign to the score if an error occurs. If 'raise', the error is raised
        :param fit_params: parameters to pass to the fit method of the estimator
        :param proba: if True, predict_proba method will be used for prediction instead of predict
        :param pre_dispatch: number of jobs to dispatch to workers. Can be used to control speed and memory
        :param n_jobs: number of jobs to run in parallel. If -1, all CPUs are used. If 1, no parallelization is used
        :param random_state: random state to use for shuffling and stratification if the dataframe is split
        :param return_train_score: if True, the training scores will be added to the returned dictionary
        :param kwargs: additional parameters to pass to the hyperparameter search method
        :return: a DataFrame containing the results of the hyperparameter search
        """
        result = {"params": [], "test_scores": []}  # TODO: decide what to put into result
        if return_train_score:
            result["train_scores"] = []
        init_params, fit_params = separate_params(kind, params, fit_params, n_iter, random_state)
        if estimator is None:
            estimator = self.model
            if estimator is None:
                raise RuntimeError('Model is not specified. Please add a model first')
        for init_param, fit_param in zip(init_params, fit_params):
            res = self.cross_validate(estimator=clone_model(estimator, init_param),
                                      cv=cv,
                                      split_kind=split_kind,
                                      sort_by=sort_by,
                                      ascending=ascending,
                                      stratify_by=stratify_by,
                                      scorers=scorers,
                                      error_score=error_score,
                                      fit_params={**kwargs, **fit_param},
                                      proba=proba,
                                      pre_dispatch=pre_dispatch,
                                      n_jobs=n_jobs,
                                      return_train_score=return_train_score,
                                      random_state=random_state)
            result["params"].append({**init_param, **fit_param})
            result["test_scores"].append(res["test_score"])
            if return_train_score:
                result["train_scores"].append(res["train_score"])
        result["scorer_names"] = res["scorer_names"]
        return pretty_results(**result)

    def disable_pipeline(self):
        """
        Disables pipeline of self
        """
        self.pipeline.disable()

    def enable_pipeline(self):
        """
        Enables pipeline of self
        """
        self.pipeline.enable()

    def clear_pipeline(self):
        """
        Deletes all elements in the pipeline
        """
        object.__setattr__(self, 'pipeline', Pipeline())

    def stream(self,
               other: Functional | Pipeline,
               strategy: str = 'complete',
               fit: bool = False,
               parameters=None,
               transfer_model: bool = True) -> Functional:
        """
        Changes self according to the provided pipeline.
        Allows three strategies:
        1. complete - changes the dataframe according to the new
        pipeline while ignoring the same transformations at the
        beginning of both pipelines
        2. add - changes the dataframe according to the new
        pipeline without ignoring any transformations. The
        previous transformations are preserved in the pipeline
        3. rewrite - changes the dataframe according the new
        pipeline without ignoring any transformations. The
        previous transformations are discarded

        :param other: Pipeline or Functional, with which this
        Functional should be transformed
        :param strategy: (str) complete, add, or rewrite
        :param fit: (bool) if True, calls fit_transform on the
        new transformations. Otherwise calls transform
        :param parameters: parameters to be passed to the pipeline
        :param transfer_model: if the model of 'other' should be
        added to the new object (shallow copy)
        :return: modified Functional
        """
        if isinstance(other, Functional):
            other = other.pipeline
        if other.is_empty():
            return self
        start = 0
        if transfer_model:
            self.pipeline.model = other.model
            self.pipeline._fit_to_numpy = other._fit_to_numpy
            self.pipeline._fit_inplace = other._fit_inplace
            self.pipeline._squeeze = other._squeeze

        if strategy == 'rewrite':
            self.clear_pipeline()
        elif strategy == 'complete':
            start = self.pipeline.find_first_different_transformation(other)
            if start == -1:
                return self  # all transformations are already there

        if fit:
            return other.fit_transform(self, parameters=parameters, start=start)
        else:
            return other.transform(self, parameters=parameters, start=start)

    def fit(self, model, *args, to_numpy=True, inplace=False, squeeze=True, **kwargs) -> None:
        """
        Fits the provided model to the data in the pipeline.

        :param model: model that will be fitted
        :param args: additional arguments to be provided to the fit
        function of the model
        :param to_numpy: if True, the data will be converted to numpy
        first
        :param inplace: if True, the data will be separated to target
        and non-target columns inplace
        :param squeeze: if True, the separated features / target will
        be squeezed of dims with length 1
        :param kwargs: additional keyword arguments to be provided to
        the fit function of the model
        :return: fitted model
        """
        return self.pipeline.fit_model(self,
                                       model,
                                       *args,
                                       to_numpy=to_numpy,
                                       inplace=inplace,
                                       squeeze=squeeze,
                                       **kwargs)

    def predict(self, model=None, to_numpy=None, inplace=None, squeeze=None, **kwargs):
        """
        Predicts the values based on a model provided or the model saved
        in the pipeline.

        :param model: if provided, this model will be used for inference
        :param to_numpy: if True, the data will be converted to numpy
        first. If None, the 'to_numpy' value of fit will be used
        :param inplace: if True, the data will be separated inplace (if
        the separation is necessary). If None, the 'inplace' value of
        fit will be used
        :param squeeze: if True, the separated features / target will
        be squeezed of dims with length 1. If None, the 'squeeze' value of
        fit will be used
        :param kwargs: additional kw arguments to be provided to predict
        function of the model
        :return: predictions
        """
        if isinstance(model, Functional):
            model = model.pipeline.model
        return self.pipeline.predict(self, model=model, to_numpy=to_numpy, inplace=inplace, proba=False,
                                     squeeze=squeeze, **kwargs)

    def predict_proba(self, model=None, to_numpy=None, inplace=None, squeeze=None, **kwargs):
        """
        Predicts the probability values based on a model provided or the
        model saved in the pipeline.

        :param model: if provided, this model will be used for inference
        :param to_numpy: if True, the data will be converted to numpy
        first. If None, the 'to_numpy' value of fit will be used
        :param inplace: if True, the data will be separated inplace (if
        the separation is necessary). If None, the 'inplace' value of
        fit will be used
        :param squeeze: if True, the separated features / target will
        be squeezed of dims with length 1. If None, the 'squeeze' value of
        fit will be used
        :param kwargs: additional kw arguments to be provided to the
        predict_proba function of the model
        :return: probability predictions
        """
        if isinstance(model, Functional):
            model = model.pipeline.model
        return self.pipeline.predict(self, model=model, to_numpy=to_numpy, inplace=inplace, proba=True,
                                     squeeze=squeeze, **kwargs)

    def evaluate(self, model=None, scorers=None, error_score: float | int | np.nan | None = np.nan,
                 to_numpy: bool | None = None, inplace: bool | None = None, proba: bool | None = False,
                 squeeze: bool | None = None, return_predictions: bool = False, **kwargs) -> dict:
        """
        Evaluates the model for provided or default scoring functions.

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
        return self.pipeline.evaluate(self, model=model, scorers=scorers, error_score=error_score,
                                      to_numpy=to_numpy, inplace=inplace, proba=proba, squeeze=squeeze,
                                      return_predictions=return_predictions, **kwargs)

    # abstract properties / methods

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def separate(self, to_numpy):
        pass
