import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# training data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# number of pixels per mnist image
pixels = 784

# number of mnist image classes
classes = 10

# learning rate
learn_rate = 0.5

# tensorflow matrix of uknown columns and 784 rows representing pixels in an image
x = tf.placeholder(tf.float32, [None, pixels])

# weights and biases
W = tf.Variable(tf.zeros([pixels, classes]))
b = tf.Variable(tf.zeros([classes]))

# implement softmax model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# labels
y_ = tf.placeholder(tf.float32, [None, classes])

# compute mean
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# specify training algorithm
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

# launch model
sess = tf.InteractiveSession()

# initialize all created variables
tf.global_variables_initializer().run()

# train over n iterations
for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# check prediction, returns list of booleans
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# check accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))














































