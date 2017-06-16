import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

# Load in data
X, Y, test_x, test_y = mnist.load_data(one_hot=True)

# Reshape data for tflearn
X = X.reshape([-1, 28, 28, 1])

# Reshape test data for tflearn
test_x = test_x.reshape([-1, 28, 28, 1])

# Model input layer
convnet = input_data(shape=[None,28,28,1],name='input')

# Convolute: input model, size, window, activation function (rectify linear)
convnet = conv_2d(convnet, 32, 2, activation='relu')

# Pool: net, window
convnet = max_pool_2d(convnet, 2)

# Convolute: input model, size, window, activation function (rectify linear)
convnet = conv_2d(convnet, 64, 2, activation='relu')

# Pool: net, window
convnet = max_pool_2d(convnet, 2)

# Fully Connected Layer: input, input size, activation function
convnet = fully_connected(convnet, 1024, activation='relu')

# Dropout neurons
convnet = dropout(convnet, 0.8)

# Output Layer (also a fully connected layer with different activation function)
convnet = fully_connected(convnet, 10, activation='softmax')

# Run Regression on convolution net: calculate loss
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, 
						loss='categorical_crossentropy', name='targets')

# Deep Neural Network
model = tflearn.DNN(convnet)

model.fit({'input':X}, {'targets':Y}, n_epoch=10, 
			validation_set=({'input':test_x}, {'targets':test_y}),
			snapshot_step=500, show_metric=True, run_id='mnist')

# Saves weights values NOT model
model.save('tflearn_conv.model')

"""
Loading weights from previous run

model.load('tflearn_conv.model')
"""

# Get Prediction
print(model.predict([test_x[1]]))