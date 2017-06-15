import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random

# centers = random.randrange(2,5)

# Make sample data
X, y = make_blobs(n_samples=15, centers=3, n_features=2)

# X = np.array([[1, 2],
# 			 [1.5, 1.8],
# 			 [5, 8],
# 			 [8, 8],
# 			 [1, 0.6],
# 			 [9, 11],
# 			 [8, 2],
# 			 [10,2],
# 			 [9, 3],])

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()

colors = 10*["g","r","c","b","k","o"]

class Mean_Shift:
	def __init__(self, radius=None, radius_norm_step=100):
		self.radius = radius
		self.radius_norm_step = radius_norm_step

	def fit(self, data):
		# Check if user specified radius
		if self.radius == None:
			# Find the center of all the data
			all_data_centroid = np.average(data, axis=0)
			# Get Magnitude from the origin
			all_data_norm = np.linalg.norm(all_data_centroid)
			self.radius = all_data_norm / self.radius_norm_step
		centroids = {}
		# Set inital centroids
		for i in range(len(data)):
			centroids[i] = data[i]
		# Define weights, reverse the list
		weights = [i for i in range(self.radius_norm_step)][::-1]
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
					distance = np.linalg.norm(featureset-centroid)
					if distance == 0:
						distance = 0.000000001
					# Decrease weight the more steps we take
					weight_index = int(distance / self.radius)
					# Set max weight value
					if weight_index > self.radius_norm_step-1:
						weight_index = self.radius_norm_step-1
					to_add = (weights[weight_index]**2)*[featureset]
					in_bandwidth += to_add
				# Recalculate the mean of centroid
				new_centroid = np.average(in_bandwidth,axis=0)
				new_centroids.append(tuple(new_centroid))
			# Get unique elements from new centroids list
			uniques = sorted(list(set(new_centroids)))
			to_pop = []
			# Get rid of useless centroids that are too close to eachother
			for i in uniques:
				for ii in uniques:
					if i == ii:
						pass
					# Check if the are from within eachothers radius
					elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
						to_pop.append(ii)
						break
			# Remove unwanted centroids
			for i in to_pop:
				try:
					uniques.remove(i)
				except:
					pass
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
		# Empty dictionary
		self.classifications = {}
		# Number of centroids after fit
		for i in range(len(self.centroids)):
			self.classifications[i] = []
		for featureset in data:
			distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
			# Find centroid with the min distance
			classification = distances.index(min(distances))
			self.classifications[classification].append(featureset)

	def predict(self, data):
		distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
		# Find centroid with the min distance
		classification = distances.index(min(distances))
		return classification

# Initialize classifier
clf = Mean_Shift()
clf.fit(X)
# Grab centroids from classifier
centroids = clf.centroids

for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=10)

# Scatter Centroids
for c in centroids:
	plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*')

plt.show()