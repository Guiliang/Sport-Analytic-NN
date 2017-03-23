import tensorflow as tf
import numpy as np
import random
from collections import deque

"""
train the home team and away team together, use a feature to represent it.
"""
feature_num = 10
ACTIONS = 12
INITIAL_EPSILON = 0.0001  # starting value of epsilon
GAMMA = 0.99  # decay rate of past observations
BATCH = 32  # size of minibatch
SPORT = "NHL"


def create_network():
    """
    define the neural network
    :return: network output
    """
    # network weights
    with tf.name_scope("Dense_Layer_first"):
        x = tf.placeholder(tf.float32, [None, feature_num], name="x")
        with tf.name_scope('weights'):
            W1 = tf.Variable(tf.zeros([feature_num, 1000]), name="W")
        with tf.name_scope('biases'):
            b1 = tf.Variable(tf.zeros([1000]), name="b")
        with tf.name_scope('Wx_plus_b'):
            y1 = tf.matmul(x, W1) + b1
        activations = tf.nn.relu(y1, name='activation')
        tf.histogram_summary('activations', activations)

    with tf.name_scope("Dense_Layer_second"):
        with tf.name_scope('weights'):
            W2 = tf.Variable(tf.zeros([1000, ACTIONS]), name="W")
        with tf.name_scope('biases'):
            b2 = tf.Variable(tf.zeros([ACTIONS]), name="b")
        with tf.name_scope('Wx_plus_b'):
            read_out = tf.matmul(activations, W2) + b2

    return x, read_out


def get_next_train_event():
    """
    retrieve next event from sport data
    :return:
    """
    state = []
    action = 1
    reward = 0
    terminal = 0
    return state, action, reward, terminal


def get_next_test_event():
    """
    retrieve next event from sport data
    :return:
    """
    state = []
    action = 1
    reward = 0
    terminal = 0
    return state, action, reward, terminal


def get_training_batch(s_t0, a_t0, t):
    """
    combine training data to a batch
    :return:
    """
    batch_return = []
    current_batch_length = 0
    if current_batch_length < 32:
        s_t1, r_t1, a_t1, terminal = get_next_train_event()
        batch_return.append((s_t0, a_t0, r_t1, s_t1, a_t1, terminal))
        current_batch_length += 1
        t += 1
        s_t0 = s_t1
        a_t0 = a_t1
    return s_t0, a_t0, batch_return, t


def train_network(x, readout, sess):
    """
    train the network
    :param x: network input placeholder
    :param readout: network output placeholder
    :param sess:
    :return:
    """
    # define the cost function
    #  a means action and y means the output?
    a = tf.placeholder("float", [None, ACTIONS])  # Inserts a placeholder for a tensor that will be always fed.
    y = tf.placeholder("float", [None])

    readout_action = tf.reduce_sum(tf.mul(readout, a),
                                   reduction_indices=1)  # Computes the sum of elements across dimensions of a tensor.
    cost = tf.reduce_mean(tf.square(y - readout_action))  # square means
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    s_t0, r_t0, a_t, _ = get_next_train_event()
    while 1:

        s_tl, a_tl, batch, t = get_training_batch(s_t0, a_t0, t)
        print ("strating training at step" + str(t))
        # get the batch variables
        s_t_batch = [d[0] for d in batch]
        a_t_batch = [d[1] for d in batch]
        r_t_batch = [d[2] for d in batch]
        s_t1_batch = [d[3] for d in batch]
        a_t1_batch = [d[4] for d in batch]

        y_batch = []
        readout_t1_batch = readout.eval(feed_dict={x: s_t1_batch})  # get Q value of s

        for i in range(0, len(batch)):
            terminal = batch[i][5]
            # if terminal, only equals reward
            if terminal:
                y_batch.append(r_t_batch[i])
                break
            else:
                y_batch.append(r_t_batch[i] + GAMMA * (readout_t1_batch[i])[a_t1_batch])

        # perform gradient step
        train_step.run(feed_dict={
            y: y_batch,
            a: a_t_batch,
            x: s_t_batch}
        )

        # update the old values
        s_t0 = s_tl
        a_t0 = a_tl
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + SPORT + '-dqn', global_step=t)

        # print info
        state = "train"
        print("TIMESTEP:", t, "\t STATE:", state)

        if terminal:
            break


def compute_state_q(x, readout):
    """
    print testing data
    :param x: network input placeholder
    :param readout:
    :return:
    """
    t = 0
    terminal = 0
    if terminal != 1:
        s_t, r_t, a_t, terminal = get_next_test_event()
        readout_t = readout.eval(feed_dict={x: s_t})
        q_t = readout_t[a_t]
        print ("time", t, "\tstate:", s_t, "\t q_value:", q_t)


def sport_start():
    sess = tf.InteractiveSession()
    x, readout = create_network()
    train_network(x, readout, sess)
