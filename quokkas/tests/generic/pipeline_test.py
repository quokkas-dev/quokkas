import numbers
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error

from quokkas.core.frames.dataframe import DataFrame
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from quokkas.qio.parsers import read_csv
from sklearn.ensemble import RandomForestRegressor


def test_pipeline_disable(df):
    df['col_0'][0] = np.nan
    with df.pipeline:
        assert not df.pipeline._enabled
        df.fillna(0, inplace=True)
    assert df.pipeline._enabled
    assert df.pipeline._transformations is None

    df['col_0'][1] = np.nan
    try:
        with df.pipeline:
            df.fillna(wrong_argument=True)
    except:
        pass
    assert df.pipeline._enabled
    assert df.pipeline._transformations is None


def test_pipeline_equality(df_pipelined):
    df_copy = df_pipelined.copy(deep=True)
    assert df_copy.pipeline._transformations is not None
    assert len(df_copy.pipeline._transformations) == 1
    assert df_copy.pipeline._transformations[0].equals(df_pipelined.pipeline._transformations[0])
    assert df_copy.pipeline.equals(df_pipelined.pipeline)
    assert df_copy.pipeline._transformations[0] is not df_pipelined.pipeline._transformations[0]


def test_pipeline_transform(df_random):
    df_parallel = df_random.copy(deep=True)
    df_random = df_random.abs()

    def square(df):
        return df ** 2

    def cube(df):
        return df ** 3

    df_random = df_random.map(cube)
    df_parallel = df_parallel.stream(df_random)
    assert df_parallel.equals(df_random)
    assert df_parallel.pipeline.equals(df_random.pipeline)

    df_parallel = df_random.copy(deep=True)
    df_random = df_random.map(square)
    df_parallel = df_parallel.stream(df_random)
    assert df_parallel.equals(df_random)
    assert df_parallel.pipeline.equals(df_random.pipeline)


def test_fit_predict(df_random_large, df_random):
    df_random_large['target'] = df_random_large['col_1'] - df_random_large['col_2']

    df_random_large = df_random_large.targetize('target').map(lambda df: df * 2).abs()
    df_random_large.fit(LinearRegression(fit_intercept=False), squeeze=False)
    res = df_random.stream(df_random_large.pipeline).predict()
    assert type(res) == np.ndarray
    assert res.shape == (df_random.shape[0], 1)

    coefs = np.zeros((1, 10))
    coefs[0][0] = 1
    coefs[0][1] = 1
    df_random_large.pipeline.model.coef_ = coefs

    res = df_random_large.pipeline.predict(df_random)
    assert type(res) == np.ndarray
    assert res.shape == (df_random.shape[0], 1)
    assert (res.squeeze() == (df_random.iloc[:, 0] + df_random.iloc[:, 1]).to_numpy()).all()


def test_evaluate(df_random_large, df_random):
    def add_target_continuous(df):
        df['target'] = df['col_1'] - df['col_2']
        return df

    def add_target_cat(df):
        df['target'] = np.random.choice([1, 2, 3, 4, 5], df.shape[0])
        return df

    add_target = [add_target_continuous, add_target_cat, add_target_continuous]
    models = [LinearRegression(fit_intercept=False), LogisticRegression(), LinearRegression()]
    scorers = [None, None, [mean_squared_error, mean_absolute_error]]
    for i in range(len(add_target)):
        df_random_large = df_random_large.map(add_target[i]).targetize('target').map(lambda df: df * 2).abs()
        df_random_large.fit(models[i], squeeze=False)
        df_random.stream(df_random_large.pipeline)

        result = df_random.evaluate(to_numpy=True, scorers=scorers[i], return_predictions=True)
        assert [key in result for key in ['scores', 'predictions', 'score_names']]
        assert all([isinstance(result['scores'][j], numbers.Number) for j in range(len(result['scores']))])
        if not isinstance(scorers[i], list):
            assert len(result["scorer_names"]) == 1
        else:
            assert len(result["scorer_names"]) == len(scorers[i])


def test_visualization(df_random):
    df_random.to_csv('test.csv')
    df = read_csv('test.csv')
    target = np.random.randint(0, 2, df.shape[0])

    def stream(df, target):
        df['target'] = target
        return df

    df = df.map(stream, target).abs().targetize('target')
    df.to_csv('test.csv')
    df.fit(LogisticRegression())

    s = "Pipeline (\n  (inception) (\n    read_csv('test.csv')\n  )\n  (transformations) (\n    (0) Map.stream(ndarray)\n    (1) Dataframe.abs()\n    (2) Dataframe.targetize('target')\n  )\n  (completion) (\n    to_csv('test.csv')\n  )\n  (model) (\n    LogisticRegression()\n  )\n)"
    assert s == df.pipeline.__repr__()
    os.remove('test.csv')
