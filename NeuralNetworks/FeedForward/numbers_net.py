# Wikipedia Data Dump
# Image Net
# Common Crawl

"""
input > weight > hidden layer 1 (activation function) > weight >
hidden layer 2 (activation function) > weights > output layer

compayer output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer... SGD, AdaGrad)

backpropogation

feed forward + backprop = epoch (cycle) 
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Multiclass optimization
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 10 classes, 0 - 9
"""
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
"""

# Begin defining model
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# numbers
n_classes = 10

# Feed 100 features at a time
batch_size = 100

# Input data: specify matrix shape
x = tf.placeholder('float', [None, 784])

# Label of the Data
y = tf.placeholder('float')

def neural_network_model(data):
	# Create tensor/array of your data using random numbers
	# Hidden Layer 1: Weight = number of features(28 * 28) * number of nodes hidden layer 1
	#				  Biases = number of nodes hidden layer 1
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	# Hidden Layer 2: Weight = number of nodes hidden layer 1 * number of nodes hidden layer 2
	# 				  Biases = number of nodes hidden layer 2
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
	# Hidden Layer 3: Weight = number of nodes hidden layer 2 * number of nodes hidden layer 3
	# 				  Biases = number of nodes hidden layer 3
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
	# Output Layer:   Weight = number of nodes hidden layer 3 * number of classes (numbers 0 - 9)
	# 				  Biases = number of classes (numbers 0 - 9)
	output_layer   = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					  'biases': tf.Variable(tf.random_normal([n_classes]))}
	# Layer Models: (input_data * weights) + biases
	# Layer 1
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	# Perform Activation Function: sigmoid
	l1 = tf.nn.relu(l1)
	# Layer 2
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	# Perform Activation Function: sigmoid
	l2 = tf.nn.relu(l2)
	# Layer 3
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	# Perform Activation Function: sigmoid
	l3 = tf.nn.relu(l3)
	# Output Layer
	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	# Return output
	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	print('Prediction: ', prediction)
	# Calculate the difference of prediction and known label
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	# Minimize the difference of prediction and known label
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	# Specify cycles of feed forward and backpropogation
	hm_epochs = 10
	# Begin TensorFlow Session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# Begin training
		for epoch in range(hm_epochs):
			epoch_loss = 0
			# Number of Cycles: Total number of samples, divide by batch size
			for _ in range(int(mnist.train.num_examples/batch_size)):
				# X: Data, Y: Labels
				# Chunk through dataset
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				# C: Cost
				# Run and optimize the cost
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			# Output where we are in the optimization process
			print('Epoch: ',epoch, ', Completed out of:', hm_epochs, ', Loss: ',epoch_loss)
		# Returns maximum value in predictions and labels, then check that they are equal
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)