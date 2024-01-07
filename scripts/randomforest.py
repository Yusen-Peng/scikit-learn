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

#encoded columns
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)

#add encoded column data
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

#finalize refactoring inputs
X_train = train_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, train_targets)

test_pred = model.predict(X_test)
test_proba = model.predict_proba(X_test)
print(pd.value_counts(test_pred))
print(test_proba)

from sklearn.metrics import accuracy_score, confusion_matrix
score = accuracy_score(test_targets, test_pred)
print(score)


#confusion matrix
from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(test_targets, test_pred, normalize='true')

sns.heatmap(confusion_mat, annot=True) # visualize it with a heatmap
plt.show()


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1, random_state=42)
model.fit(X_train, train_targets)
print(model.score(X_test, test_targets))


#plot individual decision trees
from sklearn.tree import plot_tree
plot_tree(model.estimators_[0], max_depth=2, feature_names=X_train.columns, filled=True)
plt.show()
plot_tree(model.estimators_[1], max_depth=2, feature_names=X_train.columns, filled=True)
plt.show()

#hyperparameter tuning experiment
#n_estimators (number of decision trees)
model_tuned_1 = RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=50)
model_tuned_1.fit(X_train, train_targets)
print(model_tuned_1.score(X_test, test_targets))
#tuning each individual decision tree: max_depth, max_leaf_nodes 
#(omitted)

#more hyperparameter: max_features
model_tuned_2 = RandomForestClassifier(n_jobs=-1, random_state=42, max_features=5)
model_tuned_2.fit(X_train, train_targets)
print(model_tuned_2.score(X_test, test_targets))