# Training Data: Daily Closing Price of the S&P 500
# August 2000 - August 2016
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm
import time # Helper Libraries

# Load Data
X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', 50, True)

# Normalize Data: (Price / Initial Price ) - 1
# Denormalize Data: Initial Price * (Value + 1)

# Build Sequential Model
model = Sequential()

# First Layer: LSTM Layer
model.add(LSTM(
    input_dim=1,
    output_dim=50, 			# Units in Layer
    return_sequences=True)) # Outputs of this Layer always fed into next

# Dropout 'failed' nuerons
model.add(Dropout(0.2))

# Output Layer: LSTM Layer
model.add(LSTM(
    100,					 # 100 Units in Layer
    return_sequences=False)) # NOT fed into next layer

# Dropout 'failed' neurons
model.add(Dropout(0.2))

# Aggregate output vector into one value
model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

# Get algorithm start time
start = start.time()

# Compile using specified loss function
model.compile(loss='mse', optimizer='rmsprop')

# Output time difference to user
print 'Compilation time: ', time.time() - start

# Begin training model
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=1,
    validation_split=0.05)

# Plot the predictions
predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)

"""
	Recurrent Nets can model sequential data because the
	hidden state is affected by the input AND the previous
	hidden state

	Solution to the Vanishing Gradient Problem:
		* Use LSTM(Long Short Term Memory Cells) to remember
			long-term dependencies
"""