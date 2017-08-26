import numpy as np

# Read Text File
data = open('kafka.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('Data has %d chars, %d unique' % (data_size, vocab_size))

char_to_ix = { ch:i for i, ch in enumerate(chars)}
ix_to_char = { i:ch for i, ch in enumerate(chars)}
print(char_to_ix)
print(ix_to_char)

vector_for_char_a = np.zeros((vocab_size, 1))
vector_for_char_a[char_to_ix['a']] = 1
print(vector_for_char_a.ravel())

# Define Recurrent Neural Net Model
hidden_size = 100
seq_length  = 25
learning_rate = 1e-1

# Connect vector containing one input to hidden layer
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01

# Connects hidden layer to itself. 
Whh = np.random.randn(hidden_size, hidden_size) * 0.01

# Parameters to connect the hidden layer to the output
Why = np.random.randn(vocab_size, hidden_size) * 0.01

# Hidden Bias
bh = np.zeros((hidden_size, 1))

# Output Bias
by = np.zeros((vocab_size, 1))

def loss_func(inputs, targets, hprev):
	"""
		inputs, targets are both list of integers.
		hprev is Hx1 array of initial hidden state
		returns the loss, gradients on model parameters, and last hidden state
	"""
	# Store our inputs, hidden states, outputs and probability values
	xs, hs, ys, ps, = {}, {}, {}, {} # Empty dicts
	"""
		- Each of these are going to be SEQ_LENGTH(here 25) long dicts i.e. 1 vector
			per time(seq) step
		- xs will store 1 hot encoded input characters for each of 25 time
			steps (100, 25 times) plus a -1 indexed initial state
		- To calculate the hidden state at t = 0
		- ys will store targets i.e. expected outputs for 25 times
			(26, 25 times), unnormalized probabs
		- ps will take the ys and convert them to normalzed probab for chars
		- We could have used lists BUT we need an entry with -1 to calc
			the 0th hidden layer
	"""

	# Initialize the previous hidden state
	hs[-1] = np.copy(hprev)

	# Initialize loss as 0
	loss = 0

	# Forward pass
	for t in xrange(len(inputs)):
		xs[t] = np.zeros((vocab_size, 1))








































