import pandas as pd

from .generic import _BaseExplainer


class CategoricalDetector(_BaseExplainer):
    """
    Detects categorical columns.

    :param strategy: 'count', 'type' or 'count&type'. If
    'count', the categorical features will be detected based on the
    number of distinct values (with border being min(cat_share * num_rows,
    cat_number)), if 'type' - based on the type of the column, if
    'count&type' - based on both
    :param include: if provided, only those features will be analysed
    :param exclude: if provided, these features won't be analysed
    :param cat_number: if 'count' or 'count&type', the features with fewer
    than this number of distinct values will be considered categorical (if
    cat_share * num_rows is larger than cat_number)
    :param cat_share: if 'count' or 'count&type', the features with less
    than cat_share * num_rows of distinct values will be considered categorical
    (if cat_number is larger than cat_share * num_rows)
    """

    OBJECT_KINDS = "OUSVb"
    DEFAULT_CAT_NUMBER = 20
    DEFAULT_CAT_SHARE = 0.1

    ALLOWED_STRATEGIES = ['count', 'type', 'count&type']

    def __init__(self,
                 strategy='count',
                 include=None,
                 exclude=None,
                 cat_number=DEFAULT_CAT_NUMBER,
                 cat_share=DEFAULT_CAT_SHARE):

        _BaseExplainer.__init__(self, include=include, exclude=exclude, include_target=True)
        self.cat_number = cat_number
        self.cat_share = cat_share
        self.validate_strategy(strategy)
        self.strategy = self.ALLOWED_STRATEGIES.index(strategy)  # 0: count, 1: type, 2: count&type
        self.columns = None

    def fit(self, df):
        """
        Determines categorical columns based on the strategy specified
        during initialization

        :param df: dataframe to be analysed
        """
        cols = list(self._select_columns(df))

        self.columns = CategoricalDetector.determine_categorical(df, cols, self.strategy, cat_number=self.cat_number,
                                                                 cat_share=self.cat_share)

        self.fitted = True

    @staticmethod
    def validate_strategy(strategy):
        """
        Validates that the strategy is one of the known values

        :param strategy: strategy to be validated
        """
        if strategy not in CategoricalDetector.ALLOWED_STRATEGIES:
            raise ValueError("received unknown strategy parameter"
                             f"{strategy}. Expected one of:"
                             f" {CategoricalDetector.ALLOWED_STRATEGIES}")

    @staticmethod
    def determine_categorical(df, cols, strategy, cat_number=DEFAULT_CAT_NUMBER,
                              cat_share=DEFAULT_CAT_SHARE):
        """
        The core fit function - detects categorical features within provided
        columns. The decision to make it static is based on the fact that this
        function can be reused in other classes, e.g. DistVisualizer

        :param df: dataframe to be analysed
        :param cols: columns to be analysed within dataframe
        :param strategy: strategy as defined above
        :param cat_number: cat_number as defined above
        :param cat_share: cat_share as defined above
        :return: categorical columns
        """
        columns = []

        border = CategoricalDetector.get_border(cat_number, cat_share, df.shape[0])

        for column in cols:
            if isinstance(df[column].dtype, pd.CategoricalDtype) \
                    or (strategy > 0 and df[column].dtype.kind in CategoricalDetector.OBJECT_KINDS) \
                    or ((strategy == 0 or strategy == 2) and len(pd.unique(df[column])) < border):
                columns.append(column)

        return columns

    @staticmethod
    def get_border(cat_number, cat_share, n_rows):
        """
        Calculates the decision boundary for the number of the
        unique values per feature to be / not be considered categorical

        :param cat_number: as defined above
        :param cat_share: as defined above
        :param n_rows: number of samples in the dataframe
        :return:
        """
        return min(cat_number, cat_share * n_rows)

    def visualize(self):
        raise NotImplementedError("Categorical detector doesn't support visualization")

    def __repr__(self):
        return repr(self.columns)

    def to_list(self):
        return self.columns
