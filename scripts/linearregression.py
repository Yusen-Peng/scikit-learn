from urllib.request import urlretrieve
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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

#correlation matrix
sns.heatmap(medical_df.corr(), cmap='Blues', annot=True)
plt.title('correlation matrix')
plt.show()


plt.title('Age vs. Charges')
sns.scatterplot(data=medical_df, x='age', y='charges', alpha=0.7, s=15)
plt.show()