import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
			 [1.5, 1.8],
			 [5, 8],
			 [8, 8],
			 [1, 0.6],
			 [9, 11],
			 [8, 2],
			 [10,2],
			 [9, 3],])

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()

colors = 10*["g","r","c","b","k","o"]

class Mean_Shift:
	def __init__(self, radius=4):
		self.radius = radius

	def fit(self, data):
		centroids = {}
		# Set inital centroids
		for i in range(len(data)):
			centroids[i] = data[i]
		# infinite loop
		while True:
			new_centroids = []
			# cycle through known centroids
			for i in centroids:
				# all features within bandwidth/radius of centroid i
				in_bandwidth = []
				# value of centroid were on
				centroid = centroids[i]
				for featureset in data:
					# check if feature is within the radius of centroid
					if np.linalg.norm(featureset-centroid) < self.radius:
						in_bandwidth.append(featureset)
				# Recalculate the mean of centroid
				new_centroid = np.average(in_bandwidth,axis=0)
				new_centroids.append(tuple(new_centroid))
			# Get unique elements from new centroids list
			uniques = sorted(list(set(new_centroids)))
			# Copy new instance of centroids
			prev_centroids = dict(centroids)
			# Define new centroids dictionary
			centroids = {}
			for i in range(len(uniques)):
				centroids[i] = np.array(uniques[i])
			optimized = True
			# Check for any movement
			for i in centroids:
				# compare both arrays to see if theyre equal
				if not np.array_equal(centroids[i], prev_centroids[i]):
					optimized = False
				# Break from loop if were not optimized
				if not optimized:
					break;
			# Break Loop if dataset optimized
			if optimized:
				break;
		self.centroids = centroids

	def predict(self, data):
		pass

# Initialize classifier
clf = Mean_Shift()
clf.fit(X)
# Grab centroids from classifier
centroids = clf.centroids

# Scatter Data
plt.scatter(X[:,0], X[:,1], s=150)

# Scatter Centroids
for c in centroids:
	plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*')

plt.show()
















































































