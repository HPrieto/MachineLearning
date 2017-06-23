# Dependencies
# Matrix Math
import numpy as np

# Machine Learning
import tensorflow as tf

# Plotting
import matplotlib.pyplot as plt

# Clock training time
import time

import os

# Import MNIST Data
"""
    The MNIST data is split into three parts: 55k data points of training data
    10k points of test data and 5k points of vlidation data
    Every MNIST data point has two parts: an image of a handwritten digit
    and a correspoinding label.
    We;ll call the images 'x' and the labels 'y'.
    Both the training set and test set contain images and their corresponding labels
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# Each image is 28 x 28 picles. We can interpret this as a big array of nums
n_pixels = 28 * 28

# Input to the graph -- Tensorflow's MNIST images are (1, 784) vectors
# X isn't a specific value.
# It's a placeholder, a value that we'll input when we ask TensorFlow
# To run a computation. We want to be able to input any number of MNIST images
# Each flattened into a 784-dimensional vector. We represent this as a 2-D tensor of
# floating-point numbers
X = tf.placeholder(tf.float32, shape=([None, n_pixels]))

"""
    Layer creation functions
    We could do this inline but cleaner to wrap it in respective functions
    Represent the strength of connections between units
"""


def weight_variable(shape, name):
    """
        Outputs random values from a truncated normal distribution.
        Trancated means the value is either bounded below or above (or both)
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    """
        A Variable is a modifiable tensor that lives in TensorFlow's graph of
        interacting operations. It can be used and even modified by the computation.
        For machine learning applications, one generally has the model parameters
        be variables.
    """
    return tf.Variable(initial, name=name)


"""
    Bias nodes are added to increase the flexibility of
    the model to fit the data. Specifically, it allows the
    network to fit the data when all input features are equal to 00,
    and very likely decreases the bias of the fitted values elsewhere in data space
"""


def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


"""
    Neurons in a fully connected layer have full connections to
    all activations in the previous layer, as seen in regular Neural Networks.
    Thier activations can hence be computed with a matrix multiplication followed by a
    bias offset.
"""


def FC_layer(X, W, b):
    return tf.matmul(X, W) + b


"""
    Encoder:
    Out VAE model can parse the information spread thinly over the high-dimensional
    observed world of picels, and condense the most meaningful features into a
    structured distribution over reduced (20) latent dimensions
    latent = embedded space, we just see latent used in stochastic models in papers a lot
    latent means not directly observed but are rather inferred
"""
latent_dim = 20

# Number of Neurons in input layer
h_dim = 500

# Layer 1
W_enc = weight_variable([n_pixels, h_dim], 'E_enc')
b_enc = bias_variable([h_dim], 'b_enc')

"""
    TANH activation function to replicate original model
    The TANH funciton, a.k.a. hyperbolic tangent function,
    is a rescaling of the logistic sigmoid, such that its outputs range from -1 to 1.
    TANH or sigmoid? Whatever avoids the vanishing gradient problem
    (Similar to Sigmoid but ranges from -1 to 1)
"""
h_enc = tf.nn.tanh(FC_layer(X, W_enc, b_enc))

# Layer 2
W_mu = weight_variable([h_dim, latent_dim], 'W_mu')
b_mu = bias_variable([latent_dim], 'b_mu')
mu = FC_layer(h_enc, W_mu, b_mu)  # Mean

"""
    Instead of the encoder generating a vector of real values,
    it will generate a vector of means and a vector of standard devaitions.
    For reparamterization
"""

W_logstd = weight_variable([h_dim, latent_dim], 'W_logstd')
b_logstd = bias_variable([latent_dim], 'b_logstd')
logstd = FC_layer(h_enc, W_logstd, b_logstd)

"""
    Reparamterization trick - lusts us backpropagate successfully
    since normally gradient descent expects deterministic nodes
    and we have stochastic nodes
    distribution
"""
noise = tf.random_normal([1, latent_dim])

"""
    Sample from the standard deviations (tf.exp computes exponential of x element wise)
    and add the mean
    This is our latent variable we will pass to the decoder
    ULTIMATE OUTPUT
"""
z = mu + tf.mul(noise, tf.exp(.5 * logstd))
"""
    The greater standard deviation on the noise added,
    the less information we can pass using that one variabe.
    The more efficiently we can encode the original image,
    the higher we can raise the standard deviation on our gaussian until it reaches one.
    This constraint forces the encoder to be very efficient,
    creating information-rich latent variables.
    This improves generalization, so latent variables that we either randomle generated,
    or we got from encoding non-training images, will produce a nicer result when decoded.
"""

# Decoder
# Layer 1
W_dec = weight_variable([latent_dim, h_dim], 'W_dec')
b_dec = bias_variable([h_dim], 'b_dec')

# Pass in z here (and the weights and biases we just defined)
h_dec = tf.nn.tanh(FC_layer(z, W_dec, b_dec))

"""
    Layer 2: Using the original n pixels here since that is the
    dimensionality we want to restore our data to
"""
W_reconstruct = weight_variable([h_dim, n_pixels], 'W_reconstruct')
b_reconstruct = bias_variable([n_pixels], 'b_reconstruct')

"""
    Reconstruction: A vector with only values from 0 to 1.
    Image representation after layers of optimization
"""
reconstruction = tf.nn.sigmoid(FC_layer(h_dec, W_reconstruct, b_reconstruct))

"""
    Lets define our loss function

    Variational lower bound

    Add epsilon to log to prevent numerical overflow
    Information is lost because it goes from a smaller to a larger dimensinality.
    How much information is lost? We measure this using the reconstruction log-likelihood
    This measure tells us how effectibely the decoder has learned to reconstruct
    an input image x given its latent representation z.
"""
log_likelihood = tf.reduce_sum(X * tf.log(reconstruction + 1e-9) +
                               (1 - X) * tf.log(1 - reconstruction + 1e-9), reduction_indices=1)


"""
    KL Divergence
    If the encoder outputs representations z that are different
    than those from a standard normal distribution, it will receive
    a penalty in the loss. This regularizer term means
    'k the representations z of each digit sufficiently diverse'
    If we didn't include the regularizer, the encoder could learn to cheat
    and give each datapoint a representation in a different region of Euclidean space.
"""
KL_term = -.5 * tf.reduce_sum(1 + 2 * logstd - tf.pow(mu, 2) -
                              tf.exp(2 * logstd), reduction_indices=1)

# This allows us to use stochastic gradient descent with respect to the variational parameters
variational_lower_bound = tf.reduce_mean(log_likelihood - KL_term)
optimizer = tf.train.AdadeltaOptimizer().minimize(-variational_lower_bound)

# Init all variables and start the session!
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

# Add ops to save and restore all the variables
saver = tf.train.Saver()

num_iterations = 1000000
recording_interval = 1000

# Store value for these 3 terms so we can plot them later
variational_lower_bound_array = []
log_likelihood_array = []
KL_term_array = []
iteration_array = [i * recording_interval for i in range(num_iterations / recording_interval)]
for i in range(num_iterations):
    """
        np.round to make MNIST binary
        Get girst batch (200 digits)
    """
    x_batch = np.round(mnist.train.next_batch(200)[0])
    # Run our optimizer on our data
    sess.run(optimizer, feed_dict={X: x_batch})
    if (i % recording_interval == 0):
        # Every 1K iterations record these values
        vlb_eval = variational_lower_bound.eval(feed_dict={X: x_batch})
        print("Iteration: ", i, ", Loss: ", vlb_eval)
        variational_lower_bound_array.append(vlb_eval)
        log_likelihood_array.append(np.mean(log_likelihood.eval(feed_dict={X: x_batch})))
        KL_term_array.append(np.mean(KL_term.eval(feed_dict={X: x_batch})))

plt.figure()
# For the number of iterations we had
# Plot these 3 terms
plt.plot(iteration_array, variational_lower_bound_array)
plt.plot(iteration_array, KL_term_array)
plt.plot(iteration_array, log_likelihood_array)
plt.legend(['Variational Lower Bound', 'KL divergence',
            'Log Likelihood'], bbox_to_anchor=(1.05, 1), lox=2)
plt.title('Loss per iteration')

load_model = False
if load_model:
    saver.restore(sess, os.path.join(os.getcwd(), 'Trained Bernoulli VAE'))

num_pairs = 10
image_indices = np.random.randint(0, 200, num_pairs)

# Plot 10 digits
for pair in range(num_pairs):
    # Reshaping to show original test_image
    x = np.reshape(mnist.test.images[image_indices[pair]], (1, n_pixels))
    plt.figure()
    x_image = np.reshape(x, (28, 28))
    plt.subplot(121)
    plt.imshow(x_image)
    # Reconstructed image, feed the test image to the decoder
    x_reconstruction = reconstruction.eval(feed_dict={X: x})
    # Reshape it to 28 x 28 pixels
    x_reconstruction_image = (np.reshape(x_reconstruction, (28, 28)))

    # Plot it!
    plt.subplot(122)
    plt.imshow(x_reconstruction_image)
