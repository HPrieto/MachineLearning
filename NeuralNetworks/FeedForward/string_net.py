# model data before running neural net!!!
import tensorflow as tf
from model_string_data import create_feature_sets_and_labels
import numpy as np
train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

# Begin defining model
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# numbers
n_classes = 2

# Feed 100 features at a time
batch_size = 100

# Input data: specify matrix shape
x = tf.placeholder('float', [None, len(train_x[0])])

# Label of the Data
y = tf.placeholder('float')

def neural_network_model(data):
	# Create tensor/array of your data using random numbers
	# Hidden Layer 1: Weight = number of features(len(train_x[0])) * number of nodes hidden layer 1
	#				  Biases = number of nodes hidden layer 1
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
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
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	# Minimize the difference of prediction and known label
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	# Specify cycles of feed forward and backpropogation
	hm_epochs = 10
	# Begin TensorFlow Session
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		# Begin training
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i = 0
			while i < len(train_x):
				# Get batches of data/chunks of data
				start = i
				end = i + batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				# C: Cost
				# Run and optimize the cost
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += c
				i += batch_size
			# Output where we are in the optimization process
			print('Epoch: ',epoch + 1, ', Completed out of:', hm_epochs, ', Loss: ',epoch_loss)
		# Returns maximum value in predictions and labels, then check that they are equal
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ',accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)