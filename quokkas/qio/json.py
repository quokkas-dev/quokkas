import pandas as pd
from ..utils.decorators import incept
from ..utils.general_utils import maybe_quokkanize


@incept
def loads(*args, **kwargs):
    return maybe_quokkanize(pd.loads(*args, **kwargs))


@incept
def read_json(*args, **kwargs):
    return maybe_quokkanize(pd.read_json(*args, **kwargs))


@incept
def json_normalize(*args, **kwargs):
    return maybe_quokkanize(pd.json_normalize(*args, **kwargs))


