import tensorflow as tf
import numpy as np

np.random.seed(1000)
tf.set_random_seed(1000)

def to_one_hot(y):
	one_hot = np.zeros((y.size, vocab_size))
	one_hot[np.arange(y.size), y] = 1
	return one_hot

##### Creating our training data: a random sequence of numbers
vocab_size = 10
sequence_len = 20
x_ = np.random.randint(vocab_size, size=sequence_len)
y_ = x_[1:]
x_ = x_[:-1]
x = to_one_hot(x_)
y = to_one_hot(y_)
x = np.reshape(x, (sequence_len-1, 1, vocab_size)) # for easier feeding into the model
y = np.reshape(y, (sequence_len-1, 1, vocab_size))

##### Model hyperparameters
learning_rate = 0.1
lstm_size = 20
epochs = 100

##### Defining the model
state_ph = tf.placeholder(tf.float32, [1, vocab_size], name='state')
target_ph = tf.placeholder(dtype=tf.float32, name='target')

# Setting up placeholders for our LSTM cell, hidden states. So that we can have a stateful LSTM
c_state_ph = tf.placeholder(tf.float32, shape=(1, lstm_size), name='c_state')
h_state_ph = tf.placeholder(tf.float32, shape=(1, lstm_size), name='h_state')
state_tuple = tf.contrib.rnn.LSTMStateTuple(c_state_ph, h_state_ph)

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True) # state_is_tuple is True by default anyway, but just to be explicit here
w = tf.Variable(tf.truncated_normal(shape=(lstm_size, vocab_size)))

lstm_input = tf.expand_dims(state_ph, 0) # Input to LSTM needs to be of shape [batch_size, max_time, depth]. In this case it's (1, 1, vocab_size).
lstm_out, lstm_state = tf.nn.dynamic_rnn(lstm, lstm_input, initial_state=state_tuple, dtype=tf.float32)
lstm_out = tf.reshape(lstm_out, (-1, lstm_size))
logits = tf.matmul(lstm_out, w)
prediction = tf.argmax(tf.reshape(logits, (vocab_size,)))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_ph, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

##### TRAINING
c_state = np.zeros((1, lstm_size)) # batch size of 1
h_state = np.zeros((1, lstm_size))
for e in range(epochs):
	print('epoch {}/{}'.format(e+1, epochs), end='\r')
	for i in range(len(x)):
		_, new_lstm_state = sess.run([train_op, lstm_state], {state_ph: x[i], c_state_ph: c_state, h_state_ph: h_state, target_ph: y[i]})
		c_state = new_lstm_state[0]
		h_state = new_lstm_state[1]

##### TESTING
c_state = np.zeros((1, lstm_size)) # batch size of 1
h_state = np.zeros((1, lstm_size))
state = np.reshape(x[0], (1, -1)) # first number in sequence
predictions = []
for i in range(len(x)):
	pred, new_lstm_state = sess.run([prediction, lstm_state], {state_ph: state, c_state_ph: c_state, h_state_ph: h_state})
	c_state = new_lstm_state[0]
	h_state = new_lstm_state[1]

	# convert prediction to one-hot
	state = np.zeros(vocab_size,)
	state[pred] = 1
	state = np.reshape(state, (1, vocab_size))
	predictions.append(pred)

print('\nInitial state: {}'.format(x_[0])) # x_ is in normal non one-hot notation, x is in one-hot notation
print('Ground truth : {}'.format(list(y_)))
print('Prediction   : {}'.format(predictions))
