

def test_suggest_categorical(df_random_categorical):
    df_random_categorical['chars'] = [chr(i) + chr(i + 100) for i in range(df_random_categorical.shape[0])]
    cat_cols = {'cat_num_col_0', 'cat_dt_col_0', 'cat_dt_col_1',
                'cat_num_col_2', 'cat_num_col_1', 'cat_str_col_0',
                'cat_str_col_1'}
    assert set(df_random_categorical.suggest_categorical(strategy='count')) == cat_cols

    assert set(df_random_categorical.suggest_categorical(strategy='count&type')) == cat_cols.union(['chars'])

    assert set(df_random_categorical.suggest_categorical(strategy='type')) == {'chars', 'cat_str_col_1',
                                                                               'cat_str_col_0'}
