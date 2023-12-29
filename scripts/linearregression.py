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

#linear regression with scikit-learn
#inputs: dataframe
inputs = medical_df[['age']]
#targets: array 
targets = medical_df.charges

model = LinearRegression()
model.fit(inputs, targets)

sample_predictions = model.predict(inputs)
#print(sample_predictions)

#explicit parameters
coef = model.coef_
intercept = model.intercept_