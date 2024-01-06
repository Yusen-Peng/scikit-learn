from urllib.request import urlretrieve
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor


raw_df = pd.read_csv('data/weatherAUS.csv')

#remove any rows where target value is missing
raw_df.dropna(subset=['RainTomorrow'], inplace=True)
#print(raw_df.info())


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


#split train, validation, and test datasets (time series)

year = pd.to_datetime(raw_df.Date).dt.year
train_df = raw_df[year < 2015]
validation_df = raw_df[year < 2015]
test_df = raw_df[year > 2015].dropna()

#identify input and target columns
input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'
#print(input_cols)

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

val_inputs = validation_df[input_cols].copy()
val_targets = validation_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

#select numeric columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.to_list()
categorical_cols = train_inputs.select_dtypes('object').columns.to_list()
# print(numeric_cols)
# print('----------')
# print(categorical_cols)


#deal with missing data (imputation)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean') # fill missing values with the mean of the column
#fit the imputer
imputer.fit(raw_df[numeric_cols])
#compute the statistics
print(imputer.statistics_)
#fill NaNs with average value
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols]) 
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols]) 
#print(train_inputs[numeric_cols])

print(train_inputs)


#imputation
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

print(train_inputs[numeric_cols])


#encoding for categorical data
train_inputs[categorical_cols] = train_df[categorical_cols].fillna('Unknown')
test_inputs[categorical_cols] = test_df[categorical_cols].fillna('Unknown')

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])

encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)

#finalize refactoring inputs
X_train = train_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]
