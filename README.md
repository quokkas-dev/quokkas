## quokkas

Data analysis tool that you didn't know you needed.

Quokkas is a powerful pandas-based data analysis tool. In addition to the well-known pandas functionality, it provides 
tools for data preprocessing, pipelining and explorative data analysis.

Let's have a short overview of these three pillars.

### Preprocessing

With quokkas, it is incredibly easy to scale, impute, encode (incl. dates), normalize, trim, and winsorize your data. 
These operations are not only easy to use - they are fast, robust and preserve your DataFrame structure / typing. 

For instance, if you wish to standardize your data, you can simply do it like that:

```
import quokkas as qk
df = qk.read_csv('42.csv')
df = df.scale('standard') 
# or simply df.scale('standard', inplace=True)
```

By default, each transformation has an `auto=True` parameter. This parameter ensures that the transformations are 
applied only to "relevant" columns. For instance, scale, normalize, winsorize and trim are applied only to numeric 
columns, encode is applied to columns with "few" distinct values, and impute is strategy- and type-dependent (see more 
in our preprocessing deep-dive).

Additionally, by default the transformations do not include the target column(s) - which you can set via 
`df.targetize('target_column_name')` or `df.targetize(['target_one_column_name, 'target_two_column_name'])`. You can 
change that behaviour by setting `include_target=True`, like in`df.impute('simple', strategy='default', 
include_target=True)`,   

However, the user is, of course, able to simply select the columns that they want to transform or ensure that some 
columns are not transformed:
```
# only 'investor_type' and 'country_of_origin' will be ordinally encoded
df.encode(kind='ordinal', include=['investor_type', 'country_of_origin'], inplace=True) 
# the first argument is always 'kind', so we could also use df.encode('ordinal', ...)

# all automatically selected columns except 'defcon_state' will be onehot-encoded
df.encode('onehot', exclude=['defcon_state'], inplace=True)

# all columns in the dataframe except for 'familial_status' and 'accommodation' will be robustly scaled
df.scale('robust', auto=False, exclude=['familial_status', 'accommodation'], inplace=True)
```

As you might have guessed, this column selection procedure is uniform across all data preprocessing functions. Some 
preprocessing functions have several (at times kind-dependent) additional parameters. For user convenience they are 
heavily aligned with sklearn preprocessing functionality. 

For instance, `df.scale('standard')` supports additional boolean `with_mean` and `with_std` parameters, 
`df.encode('onehot')` supports additional `categories`, `drop`, `sparse`, `dtype`, `handle_unknown`, `min_frequency` and 
`max_categories` parameters, and `df.impute('simple')` supports `strategy`, `missing_values` and `fill_value` 
parameters which can be used just like you would use respective sklearn's `StandardScaler`, `OneHotEncoder`, and 
`SimpleImputer` parameters.

But how could you use the transformed data for, say, training a model? Of course, you could transfer the data to two 
numpy arrays and go from there:
```
y = df['target_name'].to_numpy()
X = df.drop('target_name'], axis=1).to_numpy()

# which is equivalent to:
X, y = df.separate(to_numpy=True)
# if 'target_name' was targetized

model = GradientBoostingClassifier(loss='log_loss', learning_rate=5e-2)
model.fit(X, y)
```
However, quokkas provides an even easier way to fit a model to the dataframe. You just need to do the following:
```
df.fit(GradientBoostingClassifier(loss='log_loss', learning_rate=5e-2))

# now you can access the trained model via
model = df.pipeline.model

# and you can make predictions for the dataframe like that:
y_pred = df.predict()

# or, if you wanted to predict the values for another dataframe, you could use:
y_pred = df_test.predict(df.pipeline.model)
```
This example forces us to think about the following natural problem: sometimes we would like to transform a dataframe in 
exactly the same way as another dataframe. This is very easy in quokkas! Let's quickly learn how to do it:

### Pipelining

By default, quokkas pipelines (most of) the dataframe functions. That means that after each transformation, quokkas 
saves the data needed to do exactly the same transform again - which can be used, for instance, on another dataframe. 
This new dataframe can be changed via `df_new.stream(df_old)` - which finds the first operation in `df_old` that wasn't 
applied to `df_new` and applies the rest of the `df_old` pipeline to `df_new`.

Here is an example: say we have some data in a csv file, and we would like to load it, add a couple of columns, 
preprocess the data, fit a model, and evaluate the performance on the test dataset. Of course, we want to evaluate the 
performance of the model in a clean way - in particular, we would like to fit all data preprocessors (e.g. scaler) 
solely on the training data, without looking at the test data. Here is how we do it:

```
# load data
df = qk.read_csv('data.csv')

# create a couple of additional variables
df['sales_cash_ratio'] = df['sales'] / df['cash'] 
df['return'] = df['price'].pct_change()
df['return_squared'] = df['return'] ** 2

# we would like to predict the returns for the next period:
df['target_return'] = df['return'].shift(-1)

# split the data into train and test sets - convenient functionality, btw!
# default split is 80% train and 20% test
df_train, df_test = df.train_test_split(random_seed=42)

# turn on the inplace mode (default inplace=False in all functions) - strictly speaking not necessary,
# could achieve the same by writing inplace=True for each preprocessing function  
qk.Mode.inplace()
# this can be undone with qk.Mode.outplace()

# targetize the target_return column, impute the missing values for feature columns
# and drop missing values for target - note that the impute function imputes missing values for all 
# columns except target, so the only missing values left after imputing will be in the target
df_train.targetize('target_return').impute().dropna()

# scale the data robustly, winsorize the data and encode auto-detected values
# and encode dates - note that scaling and winsorization (with default auto=True) 
# does not affect non-numeric columns
df_train.scale('robust').winsorize(limits=(0.005, 0.005)).encode('onehot', drop='first').encode_dates(intrayear=True)

# fit the model
df_train.fit(RandomForestRegressor())

# change df_test exactly like we changed df_train, but without refitting of scalers / encoders
# in a scenario when you want to refit transformers on the new dataframe, you can set fit=True
# by default, stream will also copy the df_train's model
df_test.stream(df_train.pipeline)
# or, alternatively, df_test.stream(df_train)

# make predictions
preds = df_test.predict()

# evaluate the results
_, trues = df_test.separate()
mse = np.mean((preds - trues)**2)
```
So, quite easy indeed. Above we used the `train_test_split` function to split the data into a training and a test set, 
but what if we wanted to split data into multiple sets, e.g. for validation? For that we can use functions 
`train_val_test_split` and `split`. Here are some examples:    
```
df = df.sample(10000)

# default sizes for train_val_test_split are (0.8, 0.1, 0.1)
accepts parameters train_size, val_size and test_size and can infer one of them if the others are specified
df_train, df_val, df_test = df.train_val_test_split(train_size=0.7, val_size=0.1)

# the same result can be achieved with the split function
df_train, df_val, df_test = df.split(n_splits=3, sizes=(0.7, 0.1, 0.2))
# equivalent to {n_splits=3, sizes=(7000, 1000, 2000)}, {sizes=(7000, 1000, 2000)}, {n_splits=3, sizes=(7000, 1000)}
# since n_splits can be inferred from sizes and one remaining size can be inferred if n_splits is specified
# if sizes are not specified at all, split function would split the data into n_splits approximately equal parts
``` 
By default, splitter `kind` is set to `shuffled`, other options include `sequential` (rows are not shuffled before 
split), `stratified` (preserves the same proportions of examples in each class of the provided column) and `sorted` (the 
dataframe is first sorted by a provided column, then a sequential split is performed). quokkas also offers a way to 
perform a k-fold split that is often used for cross-validation:
```
df.targetize('target_return')
mse = []
for df_train, df_test in df.kfold(kind='stratified', n_splits=3):
    df_train.scale('robust').winsorize(limits=(0.005, 0.005)).encode('onehot', drop='first').encode_dates(intrayear=True)
    df_train.fit(RandomForestRegressor())
    df_test.stream(df_train)
    preds = df_test.predict()
    trues = df_test.target
    mse.append(np.mean((preds - trues)**2))
```
There is also a much easier way to perform cross-validation based on k-fold split in quokkas:
```
def mse(y_true, y_pred):
    return np.mean((y_pred - y_true)**2)
result = df.cross_validate(estimator=RandomForestRegressor(), scoring=mse, cv=3, target='target_return')
```
You can obtain the fitted models, training scores and predictions by setting `return_estimator`, 
`return_train_score` and `return_predictions` parameters to `True`. The difference between the two approaches is that 
while using the `cross_validate` function is easier, it doesn't allow us to specify transformations to be performed on 
the training and test data before fitting the model or making predictions - meaning it is still useful in most cases. 

Back to pipelining! Could we transform a completely new dataframe (say, some unlabeled data) in exactly the same way? 
Well, almost. As mentioned before, quokkas pipelines most of the dataframe operations. However, it does not keep track 
of column operations, or operations that involve other dataframes. How could we manage that?

Well, nothing is easier! Quokkas `df.map()` method allows us to pipeline any function that we might want to apply to a 
dataframe. We could use it like that:
```
def add_columns(df):
    df['sales_cash_ratio'] = df['sales'] / df['cash'] 
    df['return'] = df['price'].pct_change()
    df['return_squared'] = df['return'] ** 2
    df['target_return'] = df['return'].shift(-1)
    return df # it is critical to return the dataframe in a function that will be "mapped"

df.map(add_columns)
df.targetize('target_return').scale('robust').encode('onehot', drop='first')

df_new = qk.read_csv('test_data.csv')

# stream all the changes - now with the pipelined column operations!
df_new.stream(df)
preds = df_new.predict()
```
A trained pipeline of a dataframe can be easily saved (as long as all transformations in it can be pickled). This is 
how we can do that:
```
# save
df.pipeline.save('path.pkl')

# load
pipeline = qk.Pipeline.load('path.pkl')

# visualize the pipeline
print(pipeline)

# apply loaded pipeline to a new dataframe
df_new.stream(pipeline)
```
As discussed above, all quokkas-native preprocessing functions (i.e. encode, scale, encode_dates, impute, winsorize, 
trim) are saved in the pipeline, and an arbitrary function on a dataframe can be added to the pipeline via map. Most of 
the dataframe's member functions that transform the dataframe in some way are also added to the pipeline ("pipelined") 
automatically. Here is a list of the pipelined functions:
```
df.abs() # this function got an additional inplace parameter compared to pd implementation
df.aggregate()
df.apply()
df.applymap()
df.asfreq()
df.astype()
df.bfill()
df.clip()
df.corr()
df.cov()
df.diff()
df.drop_duplicates()
df.drop()
df.droplevel()
df.dropna()
df.explode()
df.ffill()
df.fillna()
df.filter()
df.interpolate()
df.melt()
df.pipe()
df.pivot()
df.query()
df.rename_axis()
df.rename()
df.reorder_levels()
df.replace()
df.reset_index()
df.round()
df.select_dtypes()
df.shift()
df.sort_index()
df.sort_values()
df.stack()
df.swapaxes()
df.swaplevel()
df.targetize()
df.to_datetime() # similar in functionality to pd.to_datetime), but the user provides column labels instead of dataframe
df.to_period()
df.to_timestamp()
df.transform()
df.transpose()
df.unstack()
```
The following member functions preserve the pipeline without adding themselves to it:
```
df.align()
df.append()
df.asof()
df.combine()
df.combine_first()
df.corrwith()
df.dot()
df.get()
df.iloc[]
df.join()
df.loc[]
df.mask()
df.merge()
df.reindex()
df.sample()
df.set_axis()
df.set_index()
df.update()
df.where()
df.__getitem__() # i.e. df[['column_one', 'column_two']] preserves pipeline
```
Additionally, all arithmetic operations preserve the pipeline of the left element. As you might have noted, all 
operations that require another dataframe / series to work are not pipelined. This ensures that the pipeline does not 
become too large. Of course, if the user wants to pipeline these operations, they can do it via map - as dicussed above.

If the user does not wish to pipeline a certain operation, they could turn a pipeline of the dataframe off. There are 
two principal ways to do that:
```
# disable the pipeline
df.pipeline.disable()

df.abs(inplace=True) # won't be pipelined

# enable the pipeline
df.pipeline.enable()
df.abs(inplace=True) # will be pipelined

# the pipeline can also be disabled via context manager:
with df.pipeline:
    df.abs(inplace=True) # will not be pipelined
```

Every selection operation preserves the pipeline (provided that the result of the operation is a dataframe). In 
particular, each time `df.iloc[]` is called, the pipeline of the original dataframe is copied. This makes those 
selection operations a little bit slower. There is a solution for that: quokkas provides a functionality to completely 
lock all pipeline operations via `qk.Lock.global_lock()` (which can be reversed with `qk.Lock.global_unlock()`). There 
is also a convenient context manager:
```
with qk.Lock.lock():
    df = df.scale('minmax')encode('ordinal').encode_dates(intraweek=True) # none of the operations will be pipelined
```
Note the difference: disabling the pipeline prevents transformations from being added to the pipeline, while the global 
lock prevents any operation on the pipeline. In particular, even when using operations that would generally preserve the 
pipeline, with a global lock the pipeline might not be preserved!

Now that we have discussed how the pipelining works for quokkas dataframes, we can move to the last important feature of 
this package:

### Exploratory Data Analysis
Quokkas provides some very useful (in our unbiased opinion) capabilities to help the user understand the data better. 
Some provided functions are fairly standard:
for instance, the user can visualize the correlation matrix, create scatter plots for features / target variable 
(recommended if the target values are continuous), create density plots of features for each distinct value of the 
target variable as well as plot missing values for all features (and, if necessary, target). Here is an example of the 
interface:
```
# correlation plot
df.plot_correlation(auto=True, include_target=False, absolute=True, style='seaborn')

# scatter plot, n_line_items corresponds to the number of plots in one line
df.plot_scatter(include=['col_1', 'col_2', 'col_3', 'col_4'], auto=False, n_line_items=2)

# density plot (kde)
df.plot_density(auto=True, n_line_items=3)

# missing values plot, reverse=False means that the shares of missing values will be plotted
# otherwise, shares of present values would be plotted instead
df.plot_missing_values(reverse=False, figsize=(12, 5))
```
Of course, the target for scatter and density plots should be provided to the dataframe via the targetize member 
function. An attentive reader might guess that the 'include' / 'exclude' / 'auto' logic here is the same as for the 
preprocessing functionality. By default, 'auto' is enabled, so in most cases the user does not need to provide any 
arguments at all. Every charting function in quokkas allows the user to choose the style of the chart (string 
corresponding to one of the plt.styles via style argument) and the figure size (via figsize argument). 

Additionally, quokkas provides a bit of non-standard charting functionality: the user may wish to view how the feature 
values depend on the values of the target. The function `df.plot_feature_means()` does exactly that. If the target 
variable is continuous, the user may provide an integer 'buckets' argument - the target variable values will be split
in that many quantiles and the mean of each variable will be plotted for each of the quantiles. In the case when the 
target variable is categorical, the means of feature values will be plotted for each distinct value of target variable.

The user can specify if the target variable should be considered continuous or categorical via 'kind' argument: e.g. 
`df.plot_feature_means(kind='categorical')` or `df.plot_feature_means(kind='continuous')`. By default, the kind 
parameter is set to 'default', which means that quokkas will attempt to infer the type of the target variable itself.

This plot can be quantified in a simple way as well - for that, the user can use the `df.feature_importance()` function. 
This function calculates the variance of the means of the standardized feature values among different buckets / distinct 
values, corrects that variance with the expected variance of the means of the buckets / distinct values and returns the 
share of this corrected variance for each feature.

The last EDA function that we will cover is very simple: `df.suggest_categorical(strategy=...)` suggests the features 
that should be considered categorical. It has the following strategies: 'count', 'type' and 'count&type'. If 'count' is 
selected, the decision will be based on the number of distinct feature values (if there are fewer than min(cat_number, 
cat_share*n_rows), the feature will be considered categorical, where cat_number and cat_share are parameters with 
default values 20 and 0.1). If 'type' is selected, the categorical features will be selected based on column type, and 
if 'count&type' is selected, all columns that would be selected under 'type' and 'count' strategies would be selected.

We hope that you have a lot of fun working with quokkas! If you have any issues or suggestions - please feel free to 
contact us! We will do our best to help!
