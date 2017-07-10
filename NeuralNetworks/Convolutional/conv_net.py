"""
Input => (Conv => Pool) => Fully Connected Layer => Output

Convolution: (Feature Map from Original Dataset)
				* Moving window over an image that is looking for something
			  	  and then classify the N x N window over an image

Pooling:    	* Extracting the greatest value from the featureset we created from
			  	  our convolution

Fully Connected: 'Connected' Neurons
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Multiclass optimization
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# numbers
n_classes = 10

# Feed 100 features at a time
batch_size = 128

# Input data: specify matrix shape
x = tf.placeholder('float', [None, 784])

# Label of the Data
y = tf.placeholder('float')

# Rate of neurons to keep
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    # size of window   movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolution_neural_network(x):
    # Weights Layer: (5 x 5) Convolution, 1 input, 32 outputs/features
    weights = {'w_conv1': tf.Variable(tf.random_normal([5, 5, 1,  32])),
               'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               # image: (7 x 7), features: 64
               'w_fc'	: tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'out'	: tf.Variable(tf.random_normal([1024, n_classes]))}
    # For weight outputs
    biases = {'b_conv1'	: tf.Variable(tf.random_normal([32])),
              'b_conv2'	: tf.Variable(tf.random_normal([64])),
              'b_fc'	: tf.Variable(tf.random_normal([1024])),
              'out'		: tf.Variable(tf.random_normal([n_classes]))}
    # Reshape dataset for tensorflow
    # From 724 image => (28 x 28) image
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Perform Activation Function, Convolve and Pool data
    conv1 = tf.nn.relu(conv2d(x, weights['w_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    # Perform Activation Function, Convolve and Pool convolve1
    conv2 = tf.nn.relu(conv2d(conv1, weights['w_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    # Reshape data for tensorflow
    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    # optimization function
    fc = tf.nn.relu(tf.matmul(fc, weights['w_fc']) + biases['b_fc'])
    # Dropout (Mimic Dead Neurons)
    fc = tf.nn.dropout(fc, keep_rate)
    # Calculate output f(x) = (Final Connected Layer) * (Weights) + (Biases)
    output = tf.matmul(fc, weights['out']) + biases['out']
    # Return output
    return output


def train_neural_network(x):
    prediction = convolution_neural_network(x)
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
            for _ in range(int(mnist.train.num_examples / batch_size)):
                # X: Data, Y: Labels
                # Chunk through dataset
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # C: Cost
                # Run and optimize the cost
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            # Output where we are in the optimization process
            print('Epoch: ', epoch, ', Completed out of:', hm_epochs, ', Loss: ', epoch_loss)
        # Returns maximum value in predictions and labels, then check that they are equal
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
