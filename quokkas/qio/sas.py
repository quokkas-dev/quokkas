import pandas as pd
from ..utils.decorators import incept
from ..utils.general_utils import maybe_quokkanize


@incept
def read_pickle(*args, **kwargs):
    return maybe_quokkanize(pd.read_pickle(*args, **kwargs))


@incept
def read_xml(*args, **kwargs):
    return maybe_quokkanize(pd.read_xml(*args, **kwargs))


@incept
def read_parquet(*args, **kwargs):
    return maybe_quokkanize(pd.read_parquet(*args, **kwargs))


@incept
def read_orc(*args, **kwargs):
    return maybe_quokkanize(pd.read_orc(*args, **kwargs))


@incept
def read_spss(*args, **kwargs):
    return maybe_quokkanize(pd.read_spss(*args, **kwargs))


@incept
def read_gbq(*args, **kwargs):
    return maybe_quokkanize(pd.read_gbq(*args, **kwargs))


