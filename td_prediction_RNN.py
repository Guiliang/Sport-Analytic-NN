import tensorflow as tf
import math

feature_num = 7
h_size = 512  # size of hidden layer
batch_size = 8
trace_length = 10
dropout_keep_prob = 0.7
rnn_layer = 2


def create_network_RNN_type1(rnn_type='bp_every_step'):
    """
    define the neural network
    :return: network output
    """
    rnn_input = tf.placeholder(tf.float32, [batch_size, trace_length, feature_num], name="x_1")

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size, state_is_tuple=True,
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))

    single_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=dropout_keep_prob,
                                                output_keep_prob=dropout_keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * rnn_layer, state_is_tuple=True)

    state_in = single_cell.zero_state(batch_size, tf.float32)

    rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
        inputs=rnn_input, cell=cell, dtype=tf.float32, scope=rnn_type + '_rnn')

    rnn = tf.reshape(rnn_output, shape=[batch_size, -1])

    num_layer_1 = h_size * trace_length
    num_layer_2 = trace_length
    max_sigmoid_1 = -4 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
    min_sigmoid_1 = 4 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
    W1 = tf.Variable(tf.random_uniform([num_layer_1, num_layer_2], minval=min_sigmoid_1, maxval=max_sigmoid_1),
                     name="W_1")
    b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
    y1 = tf.matmul(rnn, W1) + b1
    read_out = tf.nn.sigmoid(y1, name='activation')

    y = tf.placeholder("float", [batch_size, trace_length])

    cost = tf.reduce_mean(tf.square(y - read_out))

    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    return rnn_input, read_out, y, train_step, cost


def create_network_RNN_type2(rnn_type='bp_last_step', use_state = False):
    """
    define the neural network
    :return: network output
    """
    rnn_input = tf.placeholder(tf.float32, [batch_size, trace_length, feature_num], name="x_1")

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size, state_is_tuple=True,
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))

    single_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=dropout_keep_prob,
                                                output_keep_prob=dropout_keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * rnn_layer, state_is_tuple=True)

    state_in = single_cell.zero_state(batch_size, tf.float32)

    rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
        inputs=rnn_input, cell=cell, dtype=tf.float32, scope=rnn_type + '_rnn')

    if use_state:
        rnn_last = rnn_state
    else:
        rnn_output_trans = tf.transpose(rnn_output, [1, 0, 2])  # [trace_length, batch_length, ]
        rnn_last = rnn_output_trans[-1]

    num_layer_1 = h_size
    num_layer_2 = 1
    max_sigmoid_1 = -4 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
    min_sigmoid_1 = 4 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
    W1 = tf.Variable(tf.random_uniform([num_layer_1, num_layer_2], minval=min_sigmoid_1, maxval=max_sigmoid_1),
                     name="W_1")
    b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
    y1 = tf.matmul(rnn_last, W1) + b1
    read_out = tf.nn.sigmoid(y1, name='activation')
    readout_action = tf.reduce_sum(read_out, reduction_indices=1)

    y = tf.placeholder("float", [batch_size])

    cost = tf.reduce_mean(tf.square(y - readout_action))

    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    return rnn_input, read_out, y, train_step, cost


if __name__ == '__main__':
    # create_network_RNN_type1()
    create_network_RNN_type2()
