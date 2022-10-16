import pandas as pd
from ..utils.decorators import incept
from ..utils.general_utils import maybe_quokkanize


@incept
def read_csv(*args, **kwargs):
    return maybe_quokkanize(pd.read_csv(*args, **kwargs))


@incept
def read_fwf(*args, **kwargs):
    return maybe_quokkanize(pd.read_fwf(*args, **kwargs))


@incept
def read_table(*args, **kwargs):
    return maybe_quokkanize(pd.read_table(*args, **kwargs))


