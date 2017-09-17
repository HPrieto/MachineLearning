import tensorflow as tf

# load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape, name='W', std=0.01):
	"""
	Initialize weighted neurons for single hidden layer.
	shape: dimension of weight tensor in a single hidden layer
	name: name of tensorflow variable
	std: standard deviation for weight values
	output: tensor with slightly positive weights for ReLU neurons

	tf.truncated_normal (
		shape,				- 1-D integer tensor(shape as output tensor)
		mean=0.0,			- 0-D tensor value of dtype
		stddev=1.0,			- 0-D tensor value ot dtype
		dtype=tf.float32,	- output type
		seed=None,			- python int, used to create random values
		name=None			- name for the operation
	)
	"""
	initial = tf.truncated_normal(shape, stddev=std)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name='b', b=0.1):
	"""
	Initialize biased neurons for single hidden layer.
	shape: dimension of bias tensor in a single hidden layer
	name: name of tensorflow variable
	b: bias init value
	output: tensor with initial weights of 0.1

	tf.constant (
		value,				- A constant value(or list) of output type dtype
		dtype=None,			- type of the elements of resulting tensor
		shape=None,			- Optional dimensions of resulting tensor
		name='Const',		- optional name for the tensor
		verify_shape=False	- Boolean that enables verification of a shape of values
	)
	"""
	initial = tf.constant(b, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W):
	"""
	Slide over the image spacially, computing dot products.
	x: input matrix data
	W: weight matrix
	"""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	"""
	Makes image filter representations smaller and more manageable.
	x: image filter
	"""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# model variables/hyperparameters
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

"""
tf.placeholder(
	dtype,		- data type of elements
	shape=None,	- (optional) shape of tensor to be fed
	name=None	- name for the operation
)
IMPORTANT: This tesnor will produce an error if evaluated.
			- Its value must be fed using 'feed_dict' optional argument
				to Session.run(), Tensor.eval(), or Operation.run()
			
			Ex: 
				x = tf.placeholder(tf.float32, shape=(1024, 1024))
				y = tf.matmul(x, x)

				# WILL FAIL
				sess.run(y)

				# WILL SUCCEED
				sess.run(y, feed_dict{x: rand_array})
"""

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

"""
tf.reshape (
	tensor,		- A tensor
	shape,		- A tensor. Must be either (int32 or int64)
	name=None	- Name for the operation
)
"""

# reshape image to (width * height * rgb) or 4-D tensor
x_image = tf.reshape(x, [-1, i_height, i_width, 1])

# perform convolution: ReLU(Wx + b)
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

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

# number of neurons in fully connected layer
fc_neurons = 1024

# weights and biases for fully connected layer 1
W_fc1 = weight_variable([7 * 7 * 64, fc_neurons])
b_fc1 = bias_variable([fc_neurons])

# reshape/flatten pooling layer 2
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# activate fully connected layer 1
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout to prevent overfitting when training
keep_prob = tf.placeholder(tf.float32)

# dropout to fully connected layer 1
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

"""
Readout Layer
"""

# fully connected layer weights and biases
W_fc2 = weight_variable([fc_neurons, classes])
b_fc2 = bias_variable([classes])

# convolve fully connected layer 2/output layer
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

"""
Train and Evaluate the Model
	Differences from softmax model:
	a.) gradient descent optimizer replaced with adam optimizer
	b.) include keep_prob and feed_dict for dropout
	c.) loggin to every 100th iteration in training process
	
	- tf.Session separates the process of creating the graph and the 
		process of evaluating the graph.
"""

"""
tf.reduce_mean(
	input_tensor,				- Tensor to reduce, numeric type
	axis=None,					- Dimensions to reduce. If None, reduces ALL
	keep_dims=False,			- If true, retains reduced dimensions with length 1
	name=None,					- name for the operation
	reductions_indices=None		- the old(deprecated) name for axis
)

Output: reduced tensor
Similar to np.mean

# 'x' is [[1., 1.]
#         [2., 2.]]
tf.reduce_mean(x) ==> 1.5
tf.reduce_mean(x, 0) ==> [1.5, 1.5]
tf.reduce_mean(x, 1) ==> [1.,  2.]
"""

"""
tf.argmax(
	input,					- A tensor
	axis=None,				- A tensor
	name=None,				- name for operation
	dimension=None,			- 
	output_type=tf.int64	- (optional) data type
)

Returns tensor of type 'out_type':
	- the index with the largest value across axes of a tensor
"""

"""
tf.equal(
	x,			- A tensor
	y,			- Another tensor
	name=None	- name for the operation
)

Element-wise comparison
return tensor of type bool
"""

cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# begin training session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		batch = mnist.train.next_batch(epochs)
		if i % (epochs / 10) == 0:
			train_accuracy = accuracy.eval(feed_dict={
				x: batch[0], y_:batch[1], keep_prob: 0.5 })
			print 'step %d, training accuracy %g' % (i, train_accuracy)
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5 })

	print 'test accuracy %g' % accuracy.eval(feed_dict={
		x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
		})