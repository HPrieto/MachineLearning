from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		# Seed the random number generator, so it generates the same numbers
		# every time the program runs
		random.seed(1)
		# We model a single neuron, with 3 input connections and 1 output connection.
		# We assign random weights to a 3 x 1 matrix, with range of -1 to 1
		# and mean 0
		self.synaptic_weights = 2 * random.random((3,1)) - 1

	# The sigmoid function, which describes an s shaped curve
	# we pass the weighted sum of the inputs thorugh this function
	# to normalize them between 0 and 1
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	# Gradient of the sigmoid curve
	# Measures how confident we are of the existing weight values
	# Helps us update our prediction in the right direction
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
			# Pass the training set through our neural net
			output = self.predict(training_set_inputs)
			# Calculate the error
			error = training_set_outputs - output
			# We want to minimize the error rate
			# Multiply the error by input ad again by the gradient of the sigmoid curve
			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
			# Adjust the weights
			self.synaptic_weights += adjustment


	def predict(self, inputs):
		# Pass inputs through our neural network (single neuron)
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

	# The neural network thinks
	def think(self, inputs):
		# Pass inputs through our neural network (our single neuron)
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == '__main__':
	# Initialize Neural Network
	neural_network = NeuralNetwork()
	# Print Starting weights for our Reference
	print 'Random starting synaptic weights'
	print neural_network.synaptic_weights
	# Define Training Dataset: 4 Examples [Input: 3, Output: 1] all 1's and 0's
	training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	# T function transposes the matrix from horizontal to vertical
	# Given a new list of 1's and 0's, it'll be able to predict output of 1's or 0's
	training_set_outputs = array([[0,1,1,0]]).T
	# Train the neural network using training set.
	# Do it 10,000 times and make small adjustments each time
	# Classification since we need to know if output is 1 or 0
	neural_network.train(training_set_inputs, training_set_outputs, 10000)
	# Output new weights after training
	print 'New synaptic weights after training :'
	print neural_network.synaptic_weights
	# Test the neural network
	print 'Considering new situation [1, 0, 0] -> 0:'
	print neural_network.think(array([1,0,0]))
	print 'Considering new situation [1, 1, 1] -> 1:'
	print neural_network.think(array([1,1,1]))
	print 'Considering new situation [1, 0, 1] -> 1:'
	print neural_network.think(array([1,0,1]))
	print 'Considering new situation [0, 1, 1] -> 0:'
	print neural_network.think(array([0,1,1]))
	"""
		Samples:
		[0, 0, 1] => 0
		[1, 1, 1] => 1
		[1, 0, 1] => 1
		[0, 1, 1] => 0
	"""