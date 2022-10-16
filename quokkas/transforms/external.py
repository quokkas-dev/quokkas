import numpy as np
import pandas as pd
from copy import deepcopy as _deepcopy
from .generic import _BaseTransformer


class External(_BaseTransformer):
    """
    Processes data according to a provided transformation class.

    The transformation class must have fit_transform & transform
    or a fit & transform functions.

    For instance, one could fit an sklearn standard scaler like that:

    ext = External(StandardScaler())
    df_train = ext.fit_transform(df_train)
    df_test = ext.transform(df_test)
    """

    def __init__(self, processor, inplace=False):
        self.processor = processor
        _BaseTransformer.__init__(self)
        self.inplace = inplace
        if hasattr(processor, 'fit_transform'):
            self.has_fit_transform = True
        else:
            self.has_fit_transform = False

    def fit_transform(self, df):
        """
        Calls the fit_transform function of the underlying processor on
        the provided data

        :param df: dataframe to be transformed
        :return: transformed dataframe
        """

        original_pipeline = df.pipeline
        if self.inplace:
            with original_pipeline:
                if self.has_fit_transform:
                    self.processor.fit_transform(df)
                else:
                    self.processor.fit(df)
                    self.processor.transform(df)
            df.pipeline.add(self)

        else:
            with original_pipeline:
                if self.has_fit_transform:
                    result = self.processor.fit_transform(df)
                else:
                    self.processor.fit(df)
                    result = self.processor.transform(df)
            return self._postprocess_copy(original_pipeline, result, df)

    def transform(self, df):
        """
        Calls the transform function of the underlying processor on
        the provided data

        :param df: dataframe to be transformed
        :return: transformed dataframe
        """
        original_pipeline = df.pipeline
        if self.inplace:
            with original_pipeline:
                self.processor.transform(df)
            df.pipeline.add(self)

        else:
            with original_pipeline:
                result = self.processor.transform(df)
            return self._postprocess_copy(original_pipeline, result, df)

    def _postprocess_copy(self, original_pipeline, result, df):
        """
        Returns the provided result as a formatted dataframe.
        The result must have the same number of columns / rows as the
        original dataframe

        :param original_pipeline: the pipeline of the original data
        :param result: result of the transformation (e.g. numpy array)
        :param df: origin dataframe
        :return: formatted result
        """
        if df is result:
            result.pipeline = original_pipeline.add(self)
        else:
            if isinstance(result, np.ndarray) or isinstance(result, pd.DataFrame):
                result = df._constructor(result).__finalize__(self)
            result.pipeline = _deepcopy(original_pipeline).add(self)
            result.target = df.target
        return result

    def equals(self, other):
        return _BaseTransformer.equals(self, other) and \
               self.processor == other.processor and \
               self.has_fit_transform == other.has_fit_transform
