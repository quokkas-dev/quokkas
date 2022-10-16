import pandas as pd
from ..utils.decorators import incept
from ..utils.general_utils import maybe_quokkanize


@incept
def read_html(*args, **kwargs):
    return maybe_quokkanize(pd.read_html(*args, **kwargs))


@incept
def read_hdf(*args, **kwargs):
    return maybe_quokkanize(pd.read_hdf(*args, **kwargs))


@incept
def read_feather(*args, **kwargs):
    return maybe_quokkanize(pd.read_feather(*args, **kwargs))


@incept
def read_sas(*args, **kwargs):
    return maybe_quokkanize(pd.read_sas(*args, **kwargs))


@incept
def read_sql(*args, **kwargs):
    return maybe_quokkanize(pd.read_sql(*args, **kwargs))


@incept
def read_sql_table(*args, **kwargs):
    return maybe_quokkanize(pd.read_sql_table(*args, **kwargs))


@incept
def read_sql_query(*args, **kwargs):
    return maybe_quokkanize(pd.read_sql_query(*args, **kwargs))


@incept
def read_stata(*args, **kwargs):
    return maybe_quokkanize(pd.read_stata(*args, **kwargs))


