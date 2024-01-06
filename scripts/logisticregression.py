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

#remove missing values
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
#print(raw_df.info())


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


#data visualization
loc_rainToday = px.histogram(raw_df, x='Location', title='Location vs. raindays', color='RainToday')
#loc_rainToday.show()

temps = px.scatter(raw_df.sample(10000), title='temps', x='MinTemp', y='MaxTemp', color='RainToday')
#temps.show()

#work with samples when the dataset is large
use_sample = False

sample_fraction = 0.1

if use_sample:
    raw_df = raw_df.sample(frac=sample_fraction).copy()

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
#print(train_inputs[numeric_cols])

print(train_inputs)


#actual logistic model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear') #liblinear optimizer
#print(train_inputs[numeric_cols])

#fit the model
model.fit(train_inputs[numeric_cols], train_targets)

#coefficients (weights)
print(model.coef_.tolist())
weight_df = pd.DataFrame({
    'feature': numeric_cols,
    'weight': model.coef_.tolist()[0]
})

#bar-plot visualization
plt.figure(figsize=(10, 50))
sns.barplot(data=weight_df.sort_values('weight', ascending=False), x='weight', y='feature')
plt.show()

#make prediction and evaluate its accuracy
test_pred = model.predict(test_inputs[numeric_cols])
#print(pred)
from sklearn.metrics import accuracy_score
score = accuracy_score(test_targets, test_pred)
#print(score)

#probability prediction with a specific model class
print(model.classes_)
pred_prob = model.predict_proba(test_inputs[numeric_cols])
print(pred_prob)


#confusion matrix
from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(test_targets, test_pred, normalize='true')

sns.heatmap(confusion_mat, annot=True) # visualize it with a heatmap
plt.show()