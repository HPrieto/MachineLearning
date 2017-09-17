import tensorflow as tf

# load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# initialize session
sess = tf.InteractiveSession()

# model variables
pixels = 784
classes = 10
learn_rate = 0.5
epochs = 100

# input and output nodes for computation graph to use later on
x = tf.placeholder(tf.float32, shape=[None, pixels])
y_ = tf.placeholder(tf.float32, shape=[None, classes])

# weights and biases initialized to zero
W = tf.Variable(tf.zeros([pixels, classes]), name='W') # [784, 10]
b = tf.Variable(tf.zeros([classes, 1]), name='b') # [10, 1]

# init all tensorflow model variables at once
sess.run(tf.global_variables_initializer())

# implement regression model and get predicted y
y = tf.matmul(x, W) + b

# cost using softmax function
# cross_entropy: applies softmax to unactivated prediction and sums across all classes
# reduce_mean: takes average over all sums
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# train
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

# repeatedly train model
for _ in range(1000):
	# split training set into batches
	batch = mnist.train.next_batch(epochs)
	# train over batch
	# feed_dict replaces placeholder in computation graph
	train_step.run(feed_dict={x: batch[0], y_:batch[1]})











































