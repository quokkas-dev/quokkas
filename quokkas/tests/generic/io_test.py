from quokkas.core.pipeline.pipeline import Pipeline
from quokkas.core.frames.dataframe import DataFrame
from quokkas.qio.excel import read_excel
from quokkas.qio.json import read_json, json_normalize, loads
from quokkas.qio.parsers import read_csv, read_fwf, read_table
from quokkas.qio.sas import read_pickle, read_gbq, read_xml, read_orc, read_parquet, read_spss
from quokkas.qio.other import read_html, read_hdf, read_feather, read_stata, read_sas
from quokkas.utils.test_utils import approximately_equal
import numpy as np
import os




def power(df, n):
    return df ** n


def test_pipeline_inception(csv_path):
    df = read_csv(csv_path, index_col=0)
    assert df.pipeline._inception.origin.__name__ == read_csv.__name__
    new_df = df.pipeline.reincept()
    assert new_df.equals(df)
    new_df = df.pipeline.reincept(index_col=None)
    assert len(new_df.columns) == len(df.columns) + 1

    new_df = Pipeline.incept(read_csv, csv_path, index_col=0)
    assert new_df.equals(df)


def test_inception_completion():
    arr = np.random.normal(0, 1, (1000, 10))
    df = Pipeline.incept(arr, columns=['col_' + str(i) for i in range(10)])
    assert df.pipeline._inception.origin is arr

    df.to_csv('new_tmp_file.csv')
    assert df.pipeline._completion.func.__name__ == DataFrame.to_csv.__name__

    new_df = Pipeline.incept(read_csv, 'new_tmp_file.csv', index_col=0)
    assert ((new_df - df).abs() < 1e-7).all().all()

    parallel_df = read_csv('new_tmp_file.csv', index_col=0)
    assert approximately_equal(parallel_df, new_df)
    assert parallel_df.pipeline._inception == new_df.pipeline._inception
    assert parallel_df.pipeline.equals(new_df.pipeline)

    new_df = new_df * 2

    df.pipeline.complete(new_df)
    new_df = new_df.pipeline.reincept()
    assert ((new_df - df * 2).abs() < 1e-7).all().all()

    os.remove('new_tmp_file.csv')


def test_save_load(df_random):
    df = df_random.abs()

    df = df.map(power, 2)

    df.save('tmp_dir')
    df_loaded = DataFrame.load('tmp_dir')

    assert df_loaded.equals(df)
    assert len(df_loaded.pipeline._transformations) == 2
    assert df_loaded.pipeline._inception.origin.__name__ == 'load'
    assert df.pipeline._completion.func.__name__ == 'save'
    os.remove('tmp_dir/ndframe.pkl')
    os.remove('tmp_dir/pipeline.pkl')
    os.remove('tmp_dir/target.pkl')


def test_to_excel(df):
    path = 'excel.xlsx'
    df.to_excel(path)
    assert df.pipeline._completion.func.__name__ == 'to_excel'
    os.remove(path)


def test_save_load_functions(df_mini):
    # gbq, parquet, feather, hdf, html and xml can't be tested because they require additional imports
    # sql is not tested because it requires a connection to a database
    save_functions = {'excel': df_mini.to_excel, 'csv': df_mini.to_csv, 'pickle': df_mini.to_pickle,
                      'parquet': df_mini.to_parquet,
                      'feather': df_mini.to_feather, 'hdf': df_mini.to_hdf, 'html': df_mini.to_html,
                      'json': df_mini.to_json,
                      'stata': df_mini.to_stata,
                      'xml': df_mini.to_xml, 'gbq': df_mini.to_gbq}
    load_functions = {'excel': read_excel, 'csv': read_csv, 'pickle': read_pickle,
                      'parquet': read_parquet, 'feather': read_feather, 'hdf': read_hdf,
                      'html': read_html, 'json': read_json, 'orc': read_orc,
                      'spss': read_spss, 'table': read_table, 'stata': read_stata,
                      'xml': read_xml, 'gbq': read_gbq, 'sas': read_sas,
                      'fwf': read_fwf}
    args = {'excel': ('excel.xlsx',), 'csv': ('csv.csv',), 'pickle': ('pickle.pkl',),
            'parquet': ('parquet.parquet',), 'feather': ('feather.feather',),
            'hdf': ('hdf.hdf',), 'html': ('html.html',), 'json': ('json.json',),
            'orc': ('orc.orc',), 'spss': ('spss.sav',), 'table': ('table.csv',),
            'stata': ('stata.dta',), 'xml': ('xml.xml',), 'gbq': ('gbq.csv',),
            'sas': ('sas.sas7bdat',), 'fwf': ('fwf.csv',)}
    kwargs_save = {'excel': {}, 'csv': {}, 'pickle': {}, 'parquet': {}, 'feather': {},
                   'hdf': {'key': 'meow'},
                   'html': {}, 'json': {}, 'orc': {}, 'spss': {}, 'table': {}, 'stata': {}, 'xml': {},
                   'gbq': {}, 'sas': {}, 'fwf': {}}
    kwargs_load = {'excel': {'index_col': 0}, 'csv': {'index_col': 0}, 'pickle': {}, 'parquet': {}, 'feather': {},
                   'hdf': {},
                   'html': {}, 'json': {}, 'orc': {}, 'spss': {}, 'table': {'index_col': '0'},
                   'stata': {'index_col': 'index'}, 'xml': {},
                   'gbq': {}, 'sas': {}, 'fwf': {}}
    for func in save_functions.keys():
        if func not in ['gbq', 'feather', 'parquet', 'hdf', 'html', 'xml']:
            save_functions[func](*args[func], **kwargs_save[func])
            df_loaded = load_functions[func](*args[func], **kwargs_load[func])
            assert (df_loaded == df_mini).all().all()
            assert df_mini.pipeline._completion.func.__name__ == save_functions[func].__name__
            os.remove(args[func][0])


def test_to_sth_functions(df_mini):
    # to_markdown can't be tested because they require additional imports
    save_functions = {'markdown': df_mini.to_markdown, 'latex': df_mini.to_latex}
    for func in save_functions.keys():
        if func in ['markdown']:
            continue
        result = save_functions[func]()
        assert df_mini.pipeline._completion.func.__name__ == save_functions[func].__name__

def test_save_load_functions_pandas(df_mini):
    df_mini = df_mini.to_pandas()
    # gbq, parquet, feather, hdf can't be tested because they require additional imports
    # sql is not tested because it requires a connection to a database
    save_functions = {'excel': df_mini.to_excel, 'csv': df_mini.to_csv, 'pickle': df_mini.to_pickle,
                      'parquet': df_mini.to_parquet,
                      'feather': df_mini.to_feather, 'hdf': df_mini.to_hdf, 'html': df_mini.to_html,
                      'json': df_mini.to_json,
                      'stata': df_mini.to_stata,
                      'xml': df_mini.to_xml, 'gbq': df_mini.to_gbq}
    load_functions = {'excel': read_excel, 'csv': read_csv, 'pickle': read_pickle,
                      'parquet': read_parquet, 'feather': read_feather, 'hdf': read_hdf,
                      'html': read_html, 'json': read_json, 'orc': read_orc,
                      'spss': read_spss, 'table': read_table, 'stata': read_stata,
                      'xml': read_xml, 'gbq': read_gbq, 'sas': read_sas,
                      'fwf': read_fwf}
    args = {'excel': ('excel.xlsx',), 'csv': ('csv.csv',), 'pickle': ('pickle.pkl',),
            'parquet': ('parquet.parquet',), 'feather': ('feather.feather',),
            'hdf': ('hdf.hdf',), 'html': ('html.html',), 'json': ('json.json',),
            'orc': ('orc.orc',), 'spss': ('spss.sav',), 'table': ('table.csv',),
            'stata': ('stata.dta',), 'xml': ('xml.xml',), 'gbq': ('gbq.csv',),
            'sas': ('sas.sas7bdat',), 'fwf': ('fwf.csv',)}
    kwargs_save = {'excel': {}, 'csv': {}, 'pickle': {}, 'parquet': {}, 'feather': {},
                   'hdf': {'key': 'meow'},
                   'html': {}, 'json': {}, 'orc': {}, 'spss': {}, 'table': {}, 'stata': {}, 'xml': {},
                   'gbq': {}, 'sas': {}, 'fwf': {}}
    kwargs_load = {'excel': {'index_col': 0}, 'csv': {'index_col': 0}, 'pickle': {}, 'parquet': {},
                   'feather': {},
                   'hdf': {},
                   'html': {}, 'json': {}, 'orc': {}, 'spss': {}, 'table': {'index_col': '0'},
                   'stata': {'index_col': 'index'}, 'xml': {},
                   'gbq': {}, 'sas': {}, 'fwf': {}}
    for func in save_functions.keys():
        if func in ['gbq', 'feather', 'parquet', 'hdf', 'html', 'xml']:
            continue
        if func in ['html']:
            with open('my_file.html', 'w') as fo:
                fo.write(df_mini.to_html())
        else:
            save_functions[func](*args[func], **kwargs_save[func])
        df_loaded = load_functions[func](*args[func], **kwargs_load[func])
        df_loaded = df_loaded.to_pandas() if isinstance(df_loaded, DataFrame) else df_loaded
        df_mini = df_mini.to_pandas() if isinstance(df_mini, DataFrame) else df_mini
        assert (df_loaded == df_mini).all().all()
        os.remove(args[func][0])
