import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from statistics import mean

"""
Regression: To take continuous data and figure out a best fit line
	to that data.
	- "Model Data" (y = mx + b)
	- If you have y and x, find out what m and b is

With machine learning it boils down to Features and Labels
	Features: 			Your attributes or continuous data
	preprocessing:   	Helps with processing speed/accuracy of sig function/data standardization
	coss_validation: 	Shuffles data for preventing biased samples
	svm:			 	Support Vector Machine, helps with regression
"""

style.use('ggplot')

# Dataset
df = quandl.get('WIKI/GOOGL')

# Organized Dataset
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# High - Low Percent
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0

# Percent Change
df['PCT_CHNG'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Only Wanted Data
df = df[['Adj. Close', 'HL_PCT', 'PCT_CHNG', 'Adj. Volume']]

# Store column in variable
forecast_col = 'Adj. Close'

# Fill Not Available Values
df.fillna(-99999, inplace=True)

# Predict 10 days out
forecast_out = int(math.ceil(0.1 * len(df)))

# Shifting columns negatively
df['label'] = df[forecast_col].shift(-forecast_out)
print(forecast_out)

"""
#Print first five rows data
print('\nDataset Head:')
print(df.head())

print('\n\nDataset Tail:')
print(df.tail())
"""
# Does not return label column
# x = np.array(df.drop(['label','Adj. Close'],1))
x = np.array(df.drop(['label'], 1))

# scale x
x = preprocessing.scale(x)

# Last 30 days of data
x_lately = x[-forecast_out:]

# Prior to 30 days of data
x = x[:-forecast_out]

# Only display available values
df.dropna(inplace=True)

y = np.array(df['label'])

# Create and shuffle test data and train data
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

# Classifier using Linear Regression
clf = LinearRegression()

# Classifier using svm
# clf = svm.SVR(kernel='poly')

# train training data for predicting into the future
clf.fit(x_train, y_train)

# Save a trained set of data on a file
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

# open serialized file
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)
# print('Accuracy: ',accuracy)
"""
Never use the same data to test and train since the same data
has already been used.
"""
# Make prediction
forecast_set = clf.predict(x_lately)

print(forecast_set, accuracy, forecast_out)

# Specify that the entire column is full of NaN
df['Forecast'] = np.nan

# Figure out the last day
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# Populate dataframe with new dates
for i in forecast_set:
    """
    Iterate through forecast_set, taking each forecast and day,
    then setting those as the values in the dataframe.
    Setting the future features as NaN (Not a Number)
    """
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    """
	Takes all of the first columns, sets them to Not a Number,
	the final column is whatever 'i' is, or forecast in this case
	"""
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
print('Accuracy: ', accuracy)

"""
Linear Regression Notes:
y = m(x) + b
y: y-values
x: x-values
m: slope
b: y-intercept

How to find the slope(m):
mean(x) * mean(y) - mean(x * y)
m = -------------------------------
(mean(x))^2 - mean(x^2)

How to find y-intercept(b):
b = mean(y) - m(mean(x))
"""
