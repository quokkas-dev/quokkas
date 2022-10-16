import numbers

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score


def test_cross_validator(df_random):
    df = df_random
    df.targetize('col_1')
    scorers = [mean_squared_error, mean_absolute_error]
    cv = 3
    #  fully correct scoring (standard case)
    result = df.cross_validate(estimator=RandomForestRegressor(n_estimators=10), scorers=scorers, cv=cv)
    assert 'test_score' in result.keys()
    assert len(result['test_score']) == cv
    assert all([len(result['test_score'][i]) == len(scorers) for i in range(cv)])
    assert all([isinstance(result['test_score'][i], np.ndarray) for i in range(len(scorers))])
    assert all([all([isinstance(result['test_score'][i][j], numbers.Number) for j in
                     range(len(result['test_score'][i]))]) for i in range(len(scorers))])

    #  partially correct scoring, different target, return_train_score=True, classifier + predict_proba
    scoring_options = [[log_loss, "5"], [log_loss, accuracy_score]]
    for scorers in scoring_options:
        print(scorers)
        df['new_col'] = np.random.randint(0, 2, df.shape[0])
        result = df.cross_validate(estimator=RandomForestClassifier(n_estimators=10), scorers=scorers, cv=cv,
                                   target='new_col', return_train_score=True, proba=True, return_estimator=True)
        for score in ['test_score', 'train_score']:
            print(score)
            assert score in result.keys()
            assert len(result[score]) == cv
            assert all([len(result[score][i]) == len(scorers) for i in range(cv)])
            assert all([isinstance(result[score][i], np.ndarray) for i in range(len(scorers))])
            assert all(
                [not np.isnan(result[score][i][0]) for i in range(len(scorers))])
            assert all([np.isnan(result[score][i][1]) for i in range(len(scorers))])
            assert all(
                [all([isinstance(result[score][i][j], numbers.Number) for j in range(len(result[score][i]))]) for
                 i in range(len(scorers))])
        assert 'estimator' in result.keys()
        assert all(isinstance(result['estimator'][i], RandomForestClassifier) for i in range(cv))

    #  automatically determine scoring functions, return_predictions=True
    for predict_proba in [True, False]:
        result = df.cross_validate(estimator=RandomForestClassifier(n_estimators=10), cv=cv, target='new_col',
                                   proba=predict_proba, return_predictions=True, return_test_indices=True)
        assert 'test_score' in result.keys()
        assert len(result['test_score']) == cv
        assert all([len(result['test_score'][i]) == 1 for i in range(cv)])
        assert all([isinstance(result['test_score'][i], np.ndarray) for i in range(1)])
        assert all(
            [all([isinstance(result['test_score'][i][j], numbers.Number) for j in range(len(result['test_score'][i]))]) for
             i in range(1)])
        for key in ['predictions', 'test_indices']:
            assert key in result.keys()
            assert len(result[key]) == cv
            assert sum([len(result[key][i]) for i in range(cv)]) == df.shape[0]
            assert all([isinstance(result[key][i], np.ndarray) for i in range(cv)])


def add_target_continuous(df):
    df['target'] = df['col_1'] - df['col_2']
    return df


def add_target_cat(df):
    df['target'] = np.random.choice([1, 2, 3, 4, 5], df.shape[0])
    return df


def test_search(df_random):
    add_target = [add_target_continuous, add_target_cat, add_target_continuous]
    models = [RandomForestRegressor(max_depth=5), LogisticRegression(), LinearRegression]
    scorers = [None, None, [mean_squared_error, mean_absolute_error]]
    params = [{'min_samples_split': [2, 3, 5, 10]}, {'C': [0.1, 0.5, 1, 2, 5, 10]}, {'fit_intercept': [True, False]}]
    for kind in ['randomized', 'grid']:
        for i in range(len(add_target)):
            df_random = df_random.map(add_target[i]).targetize('target').map(lambda df: df * 2).abs()
            result = df_random.search(kind=kind, estimator=models[i], scorers=scorers[i], params=params[i])
            if 'mean_squared_error_test' in result.columns:
                assert not result['mean_squared_error_test'].isna().any()
            if 'mean_absolute_error_test' in result.columns:
                assert not result['mean_absolute_error_test'].isna().any()
            if 'log_loss_test' in result.columns:
                assert not result['log_loss_test'].isna().any()


def test_search_cv(df_random_large):
    df_random = df_random_large
    add_target = [add_target_continuous, add_target_cat, add_target_continuous]
    models = [RandomForestRegressor(max_depth=5), LogisticRegression(), LinearRegression]
    scorers = [None, None, [mean_squared_error, mean_absolute_error]]
    params = [{'min_samples_split': [2, 3, 5, 10]}, {'C': [0.1, 0.5, 1, 2, 5, 10]}, {'fit_intercept': [True, False]}]
    cv = [3, 5, 10]
    for kind in ['randomized', 'grid']:
        for i in range(len(add_target)):
            df_random = df_random.map(add_target[i]).targetize('target').map(lambda df: df * 2).abs()
            result = df_random.search_cv(kind=kind, estimator=models[i], scorers=scorers[i], params=params[i], cv=cv[i],
                                         return_train_score=True)
            for _var in ['mean_squared_error_test', 'mean_absolute_error_test', 'log_loss_test',
                         'mean_squared_error_train', 'mean_absolute_error_train', 'log_loss_train']:
                if _var in result.columns:
                    assert not result[_var].isna().any()
