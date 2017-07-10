import numpy as np
import tensorflow as tf
import os

class DNC:
	def __init__(self, input_size, output_size, seq_len, num_words=256, word_size=64, num_heads=4):
		# Define Data
		# Input Data - [[1 0] [0 1] [0 0] [0 0]]
		self.input_size = input_size # X
		# Output Data [[0 0] [0 0] [1 0] [0 1]]
		self.output_size = output_size # Y

		# Define read + write vector size
		# 10
		self.num_words = num_words # N
		# 4 Characters
		self.word_size = word_size # W

		# Define number of read + write heads
		# We could have multiple, but just 1 for simplicity
		self.num_heads = num_heads # R

		# Size of output vector from controller that defines interactions with memory matrix
		self.interface_size = num_heads * word_size + 3 * word_size + 5 * num_heads + 3

		# The actual size of the neural network input after flatenning and
		# concatenating the input vector with the previously read vectors from memory
		self.nn_input_size = num_heads * word_size + input_size

		# Size of output
		self.nn_output_size = output_size + self.interface_size

		# Gaussian normal distribution for both outputs
		self.nn_out = tf.truncated_normal([1, self.output_size], stddev=0.1)

		# Create Memory Matrix
		self.mem_mat = tf.zeros([num_words, word_size]) # N * W

		# Other Variables
		# The usage vector records which locations have been used so far
		self.usage_vec = tf.fill([num_words, 1], 1e-6) # N*1
		# A temporal link matrix records the order in which locations were written
		self.link_mat = tf.zeros([num_words,num_words]) # N*N
		# Represents degress to which last location was written to
		self.precedence_weight = tf.zeros([num_words, 1]) # N * 1

		# Read and write head variables
		self.read_weights = tf.fill([num_words, num_heads], 1e-6) # N * R
		self.write_weights = tf.fill([num_words, 1], 1e-6) # N * 1
		self.read_vecs = tf.fill([num_heads, word_size], 1e-6) # R * W

		# Network Variables
		# Gateways into the computation graph for input output pairs
		self.i_data = tf.placeholder(tf.float32, [seq_len*2, self.input_size], name='input_node')
		self.o_data = tf.placeholder(tf.float32, [seq_len*2, self.output_size], name='output_node')

		# 2 Layer FeedForwarded Network
		self.W1 = tf.Variable(tf.truncated_normal([self.nn_input_size, 32], stddev=0.1), name='layer1_weights', dtype=tf.float32)
		self.b1 = tf.Variable(tf.zeros([32]), name='layer1_bias', dtype=tf.float32)
		self.W2 = tf.Variable(tf.truncated_normal([32, self.nn_output_size], stddev=0.1), name='layer2_weights', dtype=tf.float32)
		self.b2 = tf.Variable(tf.zeros([self.nn_output_size]), name='layer2_bias', dtype=tf.float32)

		# DNC Output Weights
		self.nn_out_weights = tf.Variable(tf.truncated_normal([self.nn_output_size, self.output_size], stddev=0.1), name='net_output_weights')
		self.interface_weights = tf.Variable(tf.truncated_normal([self.nn_output_size, self.interface_size], stddev=0.1), name='interface_weights')

		self.read_vecs_out_weight = tf.Variable(tf.truncated_normal([self.num_heads*self.word_size, self.output_size], stddev=0.1), name='read_vector_weights')

	# 3 Attention Mechanisms for read/writes to memory

	#1
	# A key vector emitted by the controller is compared to the
	# content of each location in memory according to a similarity measure
	# The similarity scores determine a weighting that can be used by the read heads
	# for associative recall or by teh write head to modify an existing vector in memory
	def content_lookup(self, key, str):
		# The 12 norm of a vector is the square root of the sum of the
		# absolute values squared
		norm_mem = tf.nn.l2_normalize(self.mem_mat, 1) # N * W
		norm_key = tf.nn.l2_normalize(key, 0) # 1 * W for write or R * W for read
		# Get similarity measure betwenn both vectors, transpose before multiplication
		# (N * W, W * 1) -> N * 1 for write
		# (N * W, W * R) -> N * R for read
		sim = tf.matmul(norm_mem, norm_key, transpose_b=True)
		# Str is 1*1 or 1*R
		# Returns similarity measure
		return tf.nn.softmax(sim*str, 0) # N * 1 or N * R

	# 2
	# Retrieves the writing allocation weighting based on the usage free list
	# The 'usage' of each location is represented as a number between 0 and 1,
	# and a weighting that picks out unused locations is delivered to the write head.

	# Independent of the size and contents of the memory, meaning that
	# DNCs can be trained to solve a task using one size of memory and later
	# upgraded to a larger memory without retraining
	def allocation_weighting(self):
		# Sorted usage - the usage vector sorted ascndingly
		# The original indices of the sorted usage vector
		sorted_usage_vec, free_list = tf.nn.top_k(-1 * self.usage_vec, k=self.num_words)
		sorted_usage_vec *= -1
		cumprod = tf.cumprod(sorted_usage_vec, axis=0, exclusive=True)
		unorder = (1-sorted_usage_vec) * cumprod

		alloc_weights = tf.zeros([self.num_words])
		I = tf.constant(np.identitiy(self.num_words, dtype=np.float32))

		# For each usage vec
		for pos, idx in enumerate(tf.unstack(free_list[0])):
			# Flatten
			m = tf.squeeze(tf.slice(I, [idx, 0], [1, -1]))
			# Add to weight matrix
			alloc_weights += m * unorder[0, pos]
		# The Allocation weighting for each row in memory
		return tf.reshape(alloc_weights, [self.num_words, 1])

	# At every time step the controller receives input vector from dataset and emits output vector.

	# It also receives a set of read vectors from the memory matrix at the previous time step via
	# the read heads. Then it emits an interface vector that defines its interactions with the memory
	# at the current time step
	def step_m(self, x):
		# Reshape input
		input = tf.concat([x, tf.reshape(self.read_vecs, [1, self.num_heads * self.word_size])], 1)

		# Forward Propogation
		l1_out = tf.matmul(input,self.W1) + self.b1
		l1_act = tf.nn.tanh(l1_out)
		l2_out = tf.matmul(input,self.W2) + self.b2
		l2_act = tf.nn.tanh(l2_out)

		# Output vector
		self.nn_out = tf.matmul(l2_act, self.nn_out_weights) # (1 * eta + Y, eta + Y * Y) -> (1 * Y)
		# Interaction vector - how to interact with memory
		self.interface_vec = tf.matmul(l2_act, self.interface_weights) # (1 * eta + Y, eta + Y * eta) -> (1 * eta)

		partition = tf.constant([[0] * (self.num_heads * self.word_size) + [1] * (self.num_heads) + [2] * (self.word_size) + [3] + \
								 [4] * (self.word_size) + [5] * (self.word_size) + \
								 [6] * (self.num_heads) + [7] + [8] + [9] * (self.num_heads * 3)], dtype=tf.int32)
		# Convert interface vector into a set of read write vectors
		# using tf.dynamic_partitions(Partitions interface_vec into 10 tensors using indices from partition)
		(read_keys, read_str, write_key, write_str,
		 erase_vec, write_vec, free_gates, alloc_gate, write_gate, read_modes) = \
		 	tf.dynamic_partition(self.interface_vec, partition, 10)

		# Read Vectors
		read_keys = tf.reshape(read_keys, [self.num_heads, self.word_size]) # R * W
		read_str = 1 + tf.nn.softplus(tf.expand_dims(read_str, 0)) # 1 * R

		# Write Vectors
		write_key = tf.expand_dims(write_key, 0) # 1 * W
		# Help init our write weights
		write_str = 1 + tf.nn.softplus(tf.expand_dims(write_str, 0)) # 1 * 1
		erase_vec = tf.nn.sigmoid(tf.expand_dims(erase_vec, 0)) # 1 * W
		write_vec = tf.expand_dims(write_vec, 0) # 1 * W

		# The degree to which locations at read heads will be freed
		free_gates = tf.nn.sigmoid(tf.expand_dims(free_gates, 0)) # 1 * R
		# The fraction of writing that is being allocated in a new location
		alloc_gate = tf.nn.sigmoig(alloc_gate) # 1
		# The amount of information to be written to memory
		write_gate = tf.nn.sigmoid(write_gate) # 1
		# The softmax distribution between the three read modes (backward, forward, lookup)
		# The read heads can use gates called read modes to switch between content lookup
		# Using a read key and reading out locations either forwards or backwards
		# In the order they were written...
		read_modes = tf.nn.softmax(tf.reshape(read_modes, [3, self.num_heads])) # 3 * R

		# Used to calculate usage vector, what's available to write to?
		retention_vec = tf.reduce_prod(1 - free_gates * self.read_weights, reduction_indices=1)
		# Used to dynamically allocate memory
		self.usage_vec = (self.usage_vec + self.write_weights - self.usage_vec * self.write_weights) * retention_vec

		# Retreives the writing allocation weighting
		alloc_weights = self.allocation_weighting() # N * 1
		# Where to write to?
		write_lookup_weights = self.content_lookup(write_key, write_str) # N * 1
		# Define our write weights now that we know how much space to allocate fo rthem and where to write to
		self.write_weights = write_gate * (alloc_gate * alloc_weights + (1 - alloc_gate) * write_lookup_weights)
		# Write erase, then write to memory
		self.mem_mat = self.mem_mat* (1 - tf.matmul(self.write_weights, erase_vec)) + \
						tf.matmul(self.write_weights, write_vec)

		# As well as writing, the controller can read from multiple locations in memory.
		# Memory can be searched based on the content of each location, or the associative
		# temporal links can be followed forward and backward to recall information written
		# In sequence or in reverse. (3rd attention mechanism)

		# Updates and returns the temproal link matrix for the latest write
		# Given the precedence vector and the link matrix from previous step
		nnweight_vec = tf.matmul(self.write_weights, tf.ones([1, self.num_words])) # N * N
		self.link_mat = (1 - nnweight_vec - tf.transpose(nnweight_vec)) * self.link_mat + \
						tf.matmul(self.write_weights, self.precedence_weight, transpose_b=True)
		self.link_mat *= tf.ones([self.num_words, self.num_words]) - tf.constant(np.identity(self.num_words, dtype=np.float32))

		self.precedence_weight = (1 - tf.reduce_sum(self.write_weights, reduction_indices=0)) * \
									self.precedence_weight + self.write_weights
		# Modes - forward, backward, content lookup
		forw_w = read_modes[2]*tf.matmul(self.link_mat, self.read_weights) # (N*N,N*R)->N*R
		look_w = read_modes[1]*self.content_lookup(read_keys, read_str) # N * R
		back_w = read_modes[0]*tf.matmul(self.link_mat, self.read_weights, transpose_a=True) # N * R

		# Use them to initialize the weights
		self.read_weights = back_w + look_w + forw_w # N * R
		# Create read vectors by applying read weights to memory matrix
		self.read_vecs = tf.transpose(tf.matmul(self.mem_mat, self.read_weights, transpose_a=True)) # (W*N,N*R)^T->R*W

		# Multitply them together
		read_vec_mut = tf.matmul(tf.reshape(self.read_vecs, [1, self.num_heads * self.word_size]),
								self.read_vecs_out_weight) # (1 * RW, RW * Y) -> (1 * Y)
		# Return output + read vecs product
		return self.nn_out + read_vec_mut

	# Output list of numbers (one hot encoded) by running the step function
	def run(self):
		big_out = []
		for t, seq in enumerate(rf.unstack(self.i_data, axis=0)):
			seq = tf.expand_dims(seq, 0)
			y = self.step_m(seq)
			big_out.append(y)
		return tf.stack(big_out, axis=0)

def main(argv=None):
	# Generate the input output sequences, randomly initialized
	num_seq = 10
	seq_len = 6
	seq_width = 4
	iterations = 1000
	con = np.random.randint(0, seq_width, size=seq_len)
	seq = np.zeros((seq_len, seq_width))
	seq[np.arange(seq_len), con] = 1
	end = np.asarray([[-1]*seq_width])
	zer = np.zeros((seq_len, seq_width))

	graph = tf.Graph()

	with graph.as_default():
		# Begin Training
		with tf.Session() as sess:
			# Init the DNC
			dnc = DNC(input_size = seq_width, output_size = seq_width, seq_len = seq_len, num_words=10,
						word_size=4, num_heads=1)

			# Calculate teh predicted output
			output = tf.squeeze(dnc.run())
			# Compare prediction to reality, get loss via sigmoid cross entropy
			loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=dnc.o_data))
			# Use regularizers for each layer of the controller
			regularizers = (tf.nn.l2_loss(dnc.W1) + tf.nn.l2_loss(dnc.W2) +
							tf.nn.l2_loss(dnc.b1) + tf.nn.l2_loss(dnc.b2))
			# To help the loss convergence faster
			loss += 5e-4 * regularizers
			# Optimize the entire thing (memory + controller) using gradient descent. dope
			optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

			# Initialize input output pairs
			tf.initialize_all_variables().run()
			final_i_data = np.concatenate((seq, zer), axis=0)
			final_o_data = np.concatenate((zer, seq), axis=0)
			# For each iteration
			for i in range(0, iterations+1):
				# Feed in each input output pair
				feed_dict = {dnc.i_data: final_i_data, dnc.o_data: final_o_data}
				# Make predictions
				l, _, predictions = sess.run([loss, optimizer, output], feed_dict=feed_dict)
				if i%100==0:
					print(i,l)
				# Print Results
				print(final_i_data)
				print(final_o_data)
				print(predictions)
if __name__ == '__main__':
	tf.app.run()