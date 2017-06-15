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
			 [1, 3],
			 [8, 9],
			 [0, 3],
			 [5, 4],
			 [6, 5],])

plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

colors = 10*["g","r","c","b","k","o"]

class K_Means:
	def __init__(self, k=2, tol=0.001, max_iter=300):
		"""
		Clusters: Number of 'groups' in dataset
		Tolerance: How much each centroid will move by percent
		Max Iterations: How many times to run K_Means
		"""
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		"""
		"""
		self.centroids = {}

		# Choose first two centroids(Can also be random)
		for i in range(self.k):
			self.centroids[i] = data[i]

		# Begin optimization process(moving centroids to center)
		for i in range(self.max_iter):
			# Contains centroids and classifications
			self.classifications = {}

			for i in range(self.k):
				# Contains featuresets within values
				self.classifications[i] = []

			for featureset in data:
				# Create a list that is being populated with K number of values
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			prev_centroids =dict(self.centroids)

			for classification in self.classifications:
				# Redifine centroids with the averages
				self.centroids[classification] = np.average(self.classifications[classification],axis=0)

			optimized = True

			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				# Check if any centroids move more than the tolerance
				if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
					print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
					optimized = False

			if optimized:
				break

	def predict(self, data):
		distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

clf = K_Means()
clf.fit(X)#Train

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],
				marker="o",color="k",s=150,linewidth=5)

for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0],featureset[1],marker="x",color=color,s=150,linewidths=5)

# unknowns = np.array([[1, 3],
# 					 [8, 9],
# 					 [0, 3],
# 					 [5, 4],
# 					 [6, 5],])
# for unknown in unknowns:
# 	classification = clf.predict(unknown)
# 	plt.scatter(unknown[0], unkown[1], marker="*", color=colors[classification], s=150, linewidths=5)
plt.show()


































