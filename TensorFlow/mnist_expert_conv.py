import tensorflow as tf

# load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape, name='', std=0.01):
	"""
	shape: dimension of weight tensor in a single hidden layer
	name: name of tensorflow variable
	std: standard deviation for weight values
	output: tensor with slightly positive weights for ReLU neurons
	"""
	initial = tf.truncated_normal(shape, stddev=std)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name='', b=0.1):
	"""
	shape: dimension of bias tensor in a single hidden layer
	name: name of tensorflow variable
	b: bias init value
	output: tensor with initial weights of 0.1
	"""
	initial = tf.constant(b, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W):
	"""
	x: input matrix data
	W: weight matrix
	convolve: slide over the image spacially, computing dot products
	"""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	"""
	x: image filter
	pooling: makes image filter representations smaller and more manageable
	"""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# model variables
pixels = 784
classes = 10
learn_rate = 0.5
epochs = 100

# input and output nodes for computation graph to use later on
x = tf.placeholder(tf.float32, shape=[None, pixels])
y_ = tf.placeholder(tf.float32, shape=[None, classes])

# convolutional layer 1: weight and bias matrices
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# reshape image to (width * height * rgb)
x_image = tf.reshape(x, [-1, 28, 28, 1])
























