import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# training data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print mnist