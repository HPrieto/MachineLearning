"""
Classification: Find a model that best divides our data
Cluster: Divide data into groups

Nearest Neighbors:  Checking to see closest points to new point in data.

KNearest Neighbors: Decide what the number of 'k' is gonna be. Decide how many
		neighbors are closest to k.

Euclidian Distance: Measure distances between given point and any other point
	- Not efficient for large datasets

UCI DATASETS: archive.ics.uci.edu/ml/datasets.html
"""
import numpy as np
from sklearn import preprocessing, cross_validation, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.text')

# Replace bad data
df.replace('?', -99999, inplace=True)

# Remove irrelevant data
df.drop(['id'], 1, inplace=True)

# Define x,y
x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# Shuffle Data for Training/testing
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = svm.SVC()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)

"""
	Vectors contain:
		- Magnitude
		- Direction

	Dot Product of two Vectors:
		V1 = [A, B]
		V2 = [C, D]
		V1 * V2 == (A * C) (B * D)
	
	X, W == Vectors
	Width = (X+ - X-) * (W / magnitude(W))
"""



















