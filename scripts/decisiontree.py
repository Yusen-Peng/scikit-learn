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


#visualization the decision tree
from sklearn.tree import plot_tree, export_text
plt.figure(figsize=(10, 5))
#visual-tree version
plot_tree(model, feature_names=X_train.columns, max_depth=3, filled=True)
plt.show()

#max-depth of the decision tree
print(model.tree_.max_depth)

#textual representation
tree_text = export_text(model, max_depth=10, feature_names=list(X_train.columns))
print(tree_text)


#decision tree assign an "importance" value to each feature
importances = model.feature_importances_
#print(importances)

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False).head(10)


plt.title('feature importance')
sns.barplot(data=importance_df, x='importance', y='feature')
plt.show()


#overfitting: good in training set; bad in validation/test set
#regularization: the process of reducing overfitting
#hyperparameter tuning -- max_depth
model_tuned = DecisionTreeClassifier(max_depth=3, random_state=42)

model_tuned.fit(X_train, train_targets)

test_pred_tuned = model_tuned.predict(X_test)
print(pd.value_counts(test_pred_tuned))
plot_tree(model_tuned, feature_names=X_train.columns, max_depth=3, filled=True)
plt.show()

#hyperparameter tuning -- max_leaf_nodes
model_tuned_2 = DecisionTreeClassifier(max_leaf_nodes=128, random_state=42)

model_tuned_2.fit(X_train, train_targets)

test_pred_tuned_2 = model_tuned_2.predict(X_test)
print(pd.value_counts(test_pred_tuned_2))
plot_tree(model_tuned, feature_names=X_train.columns, max_depth=7, filled=True)
plt.show()
