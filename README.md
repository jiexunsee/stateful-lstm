# Stateful LSTM in Tensorflow

A very simple stateful LSTM implemented in Tensorflow as a test of correctness. The LSTM is given a sequence of random numbers and trained to remember it. At test time, it is given the first number in the sequence and made to regurgitate the rest of the numbers in sequence. Depending on the random seed, the sequence may be harder or easier for the LSTM to memorise and so different random seeds can lead to different performance. Usually it's able to produce the correct sequence.

Please let me know if you find any mistake in it.

## Usage
`python stateful_lstm.py`

## Output
`epoch 100/100`

`Initial state: 3`

`Ground truth : [7, 7, 0, 1, 0, 9, 8, 9, 4, 9, 4, 4, 2, 9, 8, 2, 4, 2, 5]`

`Prediction : [7, 7, 0, 1, 0, 9, 8, 9, 4, 9, 4, 4, 2, 9, 8, 2, 4, 2, 5]`
