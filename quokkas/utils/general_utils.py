from pandas import DataFrame as PDataFrame
from ..core.frames.dataframe import DataFrame


def maybe_quokkanize(df):
    """
    This function transforms the provided object into
    a quokkkas dataframe if it is pandas dataframe.

    :param df: object to be transformed if it is pd.DataFrame
    :return: "maybe" transformed object
    """

    if isinstance(df, PDataFrame):
        return DataFrame(df)
    return df
