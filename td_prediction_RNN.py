import tensorflow as tf
import math
import os
import scipy.io as sio

FEATURE_NUMBER = 12
H_SIZE = 512  # size of hidden layer
BATCH_SIZE = 8
TRACE_LENGTH = 10
DROPOUT_KEEP_PROB = 1
RNN_LAYER = 2
GAMMA = 0.99
# DATA_DIRECTORY = "/Users/liu/Desktop/sport-analytic/Data/rnn_all_match_feature"
DATA_DIRECTORY = "/home/gla68/Documents/Hockey-data/RNN-Hockey-Training-All-feature2-scale-length-3"
# DATA_DIRECTORY = "/home/gla68/Documents/Hockey-data/RNN-Hockey-Training-All"

DIR_GAMES_ALL = os.listdir(DATA_DIRECTORY)
SPORT = "NHL"
RNN_LOG_DIR = "./log_rnn_train_feature2_len10"
RNN_SAVED_NETWORK = "./saved_rnn_networks_feature2_len10"


# Download Data
# scp -r gla68@142.58.21.224:/home/gla68/Documents/Hockey-data/RNN-Hockey-Training-All/game000060 /Users/liu/Desktop/sport-analytic/Data/rnn_all_match_feature/

def create_network_RNN_type1(rnn_type='bp_every_step'):
    """
    define the neural network
    :param rnn_type:
    :return:
    :return: network output
    """
    rnn_input = tf.placeholder(tf.float32, [BATCH_SIZE, TRACE_LENGTH, FEATURE_NUMBER], name="x_1")

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=H_SIZE, state_is_tuple=True,
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))

    single_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=DROPOUT_KEEP_PROB,
                                                output_keep_prob=DROPOUT_KEEP_PROB)

    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * RNN_LAYER, state_is_tuple=True)

    state_in = single_cell.zero_state(BATCH_SIZE, tf.float32)

    rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
        inputs=rnn_input, cell=cell, dtype=tf.float32, scope=rnn_type + '_rnn')

    rnn = tf.reshape(rnn_output, shape=[BATCH_SIZE, -1])

    num_layer_1 = H_SIZE * TRACE_LENGTH
    num_layer_2 = TRACE_LENGTH
    max_sigmoid_1 = -4 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
    min_sigmoid_1 = 4 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
    w1 = tf.Variable(tf.random_uniform([num_layer_1, num_layer_2], minval=min_sigmoid_1, maxval=max_sigmoid_1),
                     name="W_1")
    b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
    y1 = tf.matmul(rnn, w1) + b1
    read_out = tf.nn.sigmoid(y1, name='activation')

    y = tf.placeholder("float", [BATCH_SIZE, TRACE_LENGTH])

    cost = tf.reduce_mean(tf.square(y - read_out))

    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    return rnn_input, read_out, y, train_step, cost


def create_network_RNN_type2(rnn_type='bp_last_step', use_state=False):
    """
    define the neural network
    :return: network output
    """
    with tf.name_scope("LSTM_layer"):
        rnn_input = tf.placeholder(tf.float32, [None, TRACE_LENGTH, FEATURE_NUMBER], name="x_1")

        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=H_SIZE, state_is_tuple=True,
                                            initializer=tf.random_uniform_initializer(-1.0, 1.0))

        single_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=DROPOUT_KEEP_PROB,
                                                    output_keep_prob=DROPOUT_KEEP_PROB)

        cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * RNN_LAYER, state_is_tuple=True)

        state_in = single_cell.zero_state(BATCH_SIZE, tf.float32)

        rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
            inputs=rnn_input, cell=cell, dtype=tf.float32, scope=rnn_type + '_rnn')

        if use_state:
            rnn_last = rnn_state
        else:
            rnn_output_trans = tf.transpose(rnn_output, [1, 0, 2])  # [trace_length, batch_size, hidden_size]
            rnn_last = rnn_output_trans[-1]

    num_layer_1 = H_SIZE
    num_layer_2 = 1
    max_sigmoid_1 = -4 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
    min_sigmoid_1 = 4 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))

    with tf.name_scope("Dense_Layer_first"):
        w1 = tf.Variable(tf.random_uniform([num_layer_1, num_layer_2], minval=min_sigmoid_1, maxval=max_sigmoid_1),
                         name="W_1")
        b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
        y1 = tf.matmul(rnn_last, w1) + b1
        read_out = tf.nn.sigmoid(y1, name='activation')

    y = tf.placeholder("float", [None])

    with tf.name_scope("cost"):
        readout_action = tf.reduce_sum(read_out, reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
    tf.summary.histogram('cost', cost)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    return rnn_input, read_out, y, train_step, cost


def get_training_batch(s_t0, state, reward, train_number, train_len):
    """
    combine training data to a batch
    :return: [last_state_of_batch, batch, time_series]
    """
    batch_return = []
    current_batch_length = 0
    while current_batch_length < BATCH_SIZE:
        s_t1 = state[train_number]
        # r_t1 = reward[train_number]
        r_t0 = reward[train_number - 1]
        train_number += 1
        if train_number + 1 == train_len:
            # batch_return.append((s_t0, r_t1, s_t1, 1))
            batch_return.append((s_t0, r_t0, s_t1, 1))
            break
        # batch_return.append((s_t0, r_t1, s_t1, 0))
        batch_return.append((s_t0, r_t0, s_t1, 0))
        current_batch_length += 1
        s_t0 = s_t1

    return s_t0, batch_return, train_number


def train_network(sess, x, read_out, y, train_step, cost, W1_print, y1_print, b1_print, W2_print, y2_print, b2_print,
                  print_parameters=False):
    """
    train the network
    :param x:
    :param sess:
    :param cost:
    :param train_step:
    :param read_out:
    :return:
    """
    game_number = 0
    global_counter = 0

    # loading network
    saver = tf.train.Saver()
    # merge = tf.merge_all_summaries()
    merge = tf.summary.merge_all()
    # train_writer = tf.train.SummaryWriter("/home/gla68/PycharmProjects/Sport-Analytic-NN/log_rnn_train_feature1_len10", sess.graph)
    # train_writer = tf.train.SummaryWriter("./log_rnn_train_feature1_len10", sess.graph)
    train_writer = tf.summary.FileWriter(RNN_LOG_DIR, sess.graph)
    sess.run(tf.global_variables_initializer())

    # checkpoint = tf.train.get_checkpoint_state("./saved_rnn_networks_feature1_len10/")
    # checkpoint = tf.train.get_checkpoint_state("./saved_rnn_networks/")
    # if checkpoint and checkpoint.model_checkpoint_path:
    #     saver.restore(sess, checkpoint.model_checkpoint_path)
    #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
    # else:
    #     print("Could not find old network weights")

    # iterate over the training data
    for i in range(0, 2):
        for dir_game in DIR_GAMES_ALL:
            if dir_game.startswith("."):  # ignore the hidden file
                continue

            game_number += 1
            game_files = os.listdir(DATA_DIRECTORY + "/" + dir_game)
            for filename in game_files:
                if filename.startswith("reward"):
                    reward_name = filename
                elif filename.startswith("state"):
                    state_name = filename

            reward = sio.loadmat(DATA_DIRECTORY + "/" + dir_game + "/" + reward_name)
            reward = reward['rnn_eward']
            reward_count = sum(reward)
            state = sio.loadmat(DATA_DIRECTORY + "/" + dir_game + "/" + state_name)
            state = state['rnn_state']
            print ("\n load file" + str(dir_game) + " success")
            print ("reward number for all trace is" + str(reward_count)) + "respectively"
            if len(state) != len(reward):
                raise Exception('state length does not equal to reward length')

            train_len = len(state)
            train_number = 0

            # start training
            s_t0 = state[0]
            r_t0 = reward[0]
            train_number += 1

            while True:

                s_tl, batch, train_number = get_training_batch(s_t0, state, reward, train_number, train_len)
                # get the batch variables
                s_t_batch = [d[0] for d in batch]
                r_t_batch = [d[1] for d in batch]
                s_t1_batch = [d[2] for d in batch]

                y_batch = []

                # debug network with W1_print, y1_print, b1_print, W2_print, y2_print, b2_print
                if print_parameters:
                    sess.run(W1_print, feed_dict={x: s_t1_batch})
                    sess.run(y1_print, feed_dict={x: s_t1_batch})
                    sess.run(b1_print, feed_dict={x: s_t1_batch})
                    sess.run(W2_print, feed_dict={x: s_t1_batch})
                    sess.run(y2_print, feed_dict={x: s_t1_batch})
                    sess.run(b2_print, feed_dict={x: s_t1_batch})

                readout_t1_batch = read_out.eval(feed_dict={x: s_t1_batch})  # get value of s

                for i in range(0, len(batch)):
                    terminal = batch[i][3]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(float(r_t_batch[i][-1]))
                        break
                    else:
                        y_batch.append(r_t_batch[i][-1] + GAMMA * ((readout_t1_batch[i]).tolist())[0])

                # perform gradient step
                [cost_out, summary_train, _] = sess.run([cost, merge, train_step], feed_dict={y: y_batch, x: s_t_batch})
                global_counter += 1
                train_writer.add_summary(summary_train, global_step=global_counter)
                # update the old values
                s_t0 = s_tl

                # print info
                if terminal or ((train_number - 1) / BATCH_SIZE) % 5 == 1:
                    print ("TIMESTEP:", train_number, "Game:", game_number)
                    print(str((max(readout_t1_batch)[0], min(readout_t1_batch)[0])))
                    print ("cost of the network is" + str(cost_out))

                if terminal:
                    # save progress after a game
                    saver.save(sess, RNN_SAVED_NETWORK+'/' + SPORT + '-game-', global_step=game_number)

                    break

    train_writer.close()


if __name__ == '__main__':
    # create_network_RNN_type1()
    if not os.path.isdir(RNN_LOG_DIR):
        os.mkdir(RNN_LOG_DIR)
    if not os.path.isdir(RNN_SAVED_NETWORK):
        os.mkdir(RNN_SAVED_NETWORK)
    sess_nn = tf.InteractiveSession()
    rnn_input_nn, read_out_nn, y_nn, train_step_nn, cost_nn = create_network_RNN_type2()
    train_network(sess_nn, rnn_input_nn, read_out_nn, y_nn, train_step_nn, cost_nn, None, None, None, None, None, None)
