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
i_width = 28
i_height = 28
pixels = 784
classes = 10
learn_rate = 0.5
epochs = 100
filter_d = 5
stride = 1
padding = 1
output_channels = 32

# input and output nodes for computation graph to use later on
x = tf.placeholder(tf.float32, shape=[None, pixels])
y_ = tf.placeholder(tf.float32, shape=[None, classes])

"""
Convolutional Neural Network Layer 1
	- output filter image sized: (14 x 14)
"""

# convolutional layer 1: weight and bias matrices (patch cols, patch rows, input channels, output channels)
W_conv1 = weight_variable([filter_d, filter_d, 1, output_channels])
b_conv1 = bias_variable([output_channels]) # output channels

# reshape image to (width * height * rgb) or 4-D tensor
x_image = tf.reshape(x, [-1, i_height, i_width, 1])

# perform convolution: ReLU(Wx + b)
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

# perform pool: outputs 14 x 14 image filter
h_pool1 = max_pool_2x2(h_conv1)


"""
Convolutional Neural Network Layer 2
	- output filter image sized: (7 x 7)
"""

# convolutional layer 2 weights and biases
W_conv2 = weight_variable([filter_d, filter_d, output_channels, output_channels*2])
b_conv2 = bias_variable([output_channels*2])

# convolve layer 2
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# pool layer 2: outputs image of dimension (7 * 7)
h_pool2 = max_pool_2x2(h_conv2)

"""
Convolutional Neural Network Fully connected Layer 1
	* 1024 neurons to allow processing of the entire image
	a.) Reshape tensor from pooling layer into batch of vectors
	b.) multiply by a weight matrix
	c.) add a bias
	d.) apply ReLU
"""
fc_neurons = 1024

# weights and biases for fully connected layer 1
W_fc1 = weight_variable([7 * 7 * 64, fc_neurons])
b_fc1 = bias_variable([fc_neurons])

# reshape/flatten pooling layer 2
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# activate fully connected layer 1
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

























