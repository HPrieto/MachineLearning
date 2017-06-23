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
from tensorflow.contrib import rnn
# Multiclass optimization
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# numbers
n_classes = 10

# Feed 100 features at a time
batch_size = 128

# Specify cycles of feed forward and backpropogation
hm_epochs = 3

# Size of array to feed into recurrent neural network
chunk_size = 28

# How many total batches of data
n_chunks = 28

# Size of recurrent neural net
rnn_size = 128

# Input data: specify matrix shape
x = tf.placeholder('float', [None, n_chunks, chunk_size])

# Label of the Data
y = tf.placeholder('float')


def recurrent_neural_network(x):
    # Create tensor/array of your data using random numbers
    # Hidden Layer 1: Weight = number of features(28 * 28) * number of nodes hidden layer 1
    # Biases = number of nodes hidden layer 1
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    # Format data for tensorflow
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = tf.nn.rnn.BasicLSTMCell(rnn_size)
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
    # Output Layer
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    # Return output
    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    print('Prediction: ', prediction)
    # Calculate the difference of prediction and known label
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    # Minimize the difference of prediction and known label
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # Begin TensorFlow Session
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # Begin training
        for epoch in range(hm_epochs):
            epoch_loss = 0
            # Number of Cycles: Total number of samples, divide by batch size
            for _ in range(int(mnist.train.num_examples / batch_size)):
                # X: Data, Y: Labels
                # Chunk through dataset
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # Reshape epoch_x
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                # C: Cost
                # Run and optimize the cost
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            # Output where we are in the optimization process
            print('Epoch: ', epoch, ', Completed out of:', hm_epochs, ', Loss: ', epoch_loss)
        # Returns maximum value in predictions and labels, then check that they are equal
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images.reshape(
            (-1, n_chunks, chunk_size)), y: mnist.test.labels}))


train_neural_network(x)
