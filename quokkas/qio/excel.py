import pandas as pd
from ..utils.decorators import incept
from ..utils.general_utils import maybe_quokkanize


@incept
def read_excel(*args, **kwargs):
    return maybe_quokkanize(pd.read_excel(*args, **kwargs))


