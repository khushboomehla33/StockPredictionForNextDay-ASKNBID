
#Prediction of stock prices for the next day

    '''
        Real world/Buisness objectives and contraints
        * No low-latency requirement
        * Interpretaibility is important
        * Errors can be very costly
    '''
#IMPORTING LIBRARIES
    
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


#Displaying and Understanding all the datasets


prices=pd.read_csv("C:\\Users\\Rajat\\Downloads\\prices.csv")
prices.head()
prices.shape
prices['symbol'].value_counts()

prices_split=pd.read_csv("C:\\Users\\Rajat\\Downloads\\prices-split-adjusted.csv")
prices_split.head()
prices_split.shape

security=pd.read_csv("C:\\Users\\Rajat\\Downloads\\securities.csv")
security.head()
security.shape
security['GICS Sector'].value_counts()

fundamentals=pd.read_csv("C:\\Users\\Rajat\\Downloads\\fundamentals.csv")
fundamentals.head()
fundamentals.shape


#Selecting some point from full dataset for better visualization

prices1=prices[0:250]
prices2=prices_split[0:250]


prices1['date'] = pd.to_datetime(prices1.date,format='%Y-%m-%d')
prices1.index = prices1['date']


#Plotting Time-Series Graph for features of the dataset


plt.figure(figsize=(16,8))
plt.plot(prices1['open'], label='Open for prices')
plt.legend()
plt.plot(prices1['close'], label='close for prices')
plt.legend()


#Analysing the Relation among features from the dataset


prices1['date'] = pd.to_datetime(prices1.date,format='%Y-%m-%d')
prices1.index = prices1['date']

plt.figure(figsize=(16,8))
plt.plot(prices1['low'], label='low for prices')
plt.legend()
plt.plot(prices1['high'], label='high for prices')
plt.legend()


#setting index as date
prices2['date'] = pd.to_datetime(prices2.date,format='%Y-%m-%d')
prices2.index = prices2['date']

#plot
plt.figure(figsize=(16,8))
plt.plot(prices2['close'], label='Close Price history')
plt.legend()
plt.plot(prices2['open'], label='Open Price history')
plt.legend()
plt.plot(prices2['low'], label='low Price history')
plt.legend()
plt.plot(prices2['high'], label='high Price history')
plt.legend()



# Doing Standardisation

scalar=StandardScaler()
stan_data_prices=scalar.fit_transform(prices[['open','close','low','high']])
stan_data_prices_split=scalar.fit_transform(prices_split[['open','close','low','high']])

print(stan_data_prices)

print(stan_data_prices_split)

f2=fundamentals.drop(['Ticker Symbol','Period Ending'], axis = 1) 

stan_fund=scalar.fit_transform(f2)

print(stan_fund)

x1=prices.drop(['date','symbol'], axis = 1)

X=x1.iloc[1:,:-1].values
y=prices.iloc[:,-1].values



# LINEAR REGRESSION ON PRICES(.CSV) DATASET

X=stan_data_prices
print(X)

# Splitting dataset into Training and Testing data.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


# Importing Model from Sk-learn

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print(y_pred.shape)
print(X_test.shape)


# Graph Plotting

plt.scatter(y_test,y_pred)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show

#Final

y_pred

'''THANK-YOU'''







