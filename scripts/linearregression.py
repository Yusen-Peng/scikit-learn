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

"""
download the data.

"""
# medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
# urlretrieve(medical_charges_url, '../data/medical.csv')


medical_df = pd.read_csv('data/medical.csv').iloc[:, [0, 2, 3, 6]]



sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

#visualization: correlation matrix
sns.heatmap(medical_df.corr(), cmap='Blues', annot=True)
plt.title('correlation matrix')
plt.show()


plt.title('Age vs. Charges')
sns.scatterplot(data=medical_df, x='age', y='charges', alpha=0.7, s=15)
plt.show()


#Loss/Cost function
#root mean sqaured error
def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))

#linear regression with scikit-learn
#inputs: dataframe
inputs = medical_df[['age', 'bmi']]
#targets: array 
targets = medical_df.charges

model = LinearRegression()
model.fit(inputs, targets)

sample_predictions = model.predict(inputs)
#print(sample_predictions)

#cost function
cost = rmse(targets, sample_predictions)

#explicit parameters
# an array of coefficients
coef = model.coef_
#intercept
intercept = model.intercept_


#categorical features
complete_df = pd.read_csv('data/medical.csv')

sns.barplot(data=complete_df, x='smoker', y='charges')
#plt.show()


#approach 1: binary categories
smoker_code = {'no': 0, 'yes': 1}
complete_df['smoker_code'] = complete_df.smoker.map(smoker_code)

inputs = complete_df[['age', 'bmi', 'children', 'smoker_code']]
targets = complete_df.charges

model = LinearRegression()
model.fit(inputs, targets)

sample_predictions = model.predict(inputs)
#print(sample_predictions)

#cost/loss function
loss = rmse(targets, sample_predictions)
#print(model.coef_, cost)

#approach 2: one-hot encoding
from sklearn import preprocessing

#create the encoder and fit the value
enc = preprocessing.OneHotEncoder()
enc.fit(complete_df[['region']])

#encoded array
one_hot = enc.transform(complete_df[['region']]).toarray()
#print(one_hot)

#add the encoded array to the dataframe
complete_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot

#feature scaling
from sklearn.preprocessing import StandardScaler
#create the scaler and fit the value
scaler = StandardScaler()
scaler.fit(medical_df[['age', 'bmi', 'children']])

#scale the inputs
scaled_inputs = scaler.transform(medical_df[['age', 'bmi', 'children']]) 


#create a test set by splitting the dataset
from sklearn.model_selection import train_test_split
inputs_train, inputs_test, targets_train, targets_test = train_test_split(medical_df)
