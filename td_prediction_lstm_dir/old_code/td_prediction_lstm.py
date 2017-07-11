import tensorflow as tf
import math
import os
import scipy.io as sio
import traceback
import numpy as np

MAX_TRACE_LENGTH = 10
FEATURE_NUMBER = 25
BATCH_SIZE = 16
GAMMA = 1
H_SIZE = 512
# DROPOUT_KEEP_PROB = 0.8
RNN_LAYER = 2
USE_HIDDEN_STATE = False
FEATURE_TYPE = 5
ITERATE_NUM = 25

SPORT = "NHL"
REWARD_TYPE = "NEG_REWARD_GAMMA1_V3"
DATA_STORE = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature" + str(
    FEATURE_TYPE) + "-scale-neg_reward_length-dynamic"
LOG_DIR = "/cs/oschulte/Galen/models/hybrid_sl_log_NN/log_train_feature" + str(FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM)
SAVED_NETWORK = "/cs/oschulte/Galen/models/hybrid_sl_saved_NN/saved_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM)
DIR_GAMES_ALL = os.listdir(DATA_STORE)
number_of_total_game = len(DIR_GAMES_ALL)


class td_prediction_lstm:
    def __init__(self, rnn_type='bp_last_step'):
        """
        define the neural network
        :return: network output
        """
        with tf.name_scope("LSTM_layer"):
            self.rnn_input = tf.placeholder(tf.float32, [None, MAX_TRACE_LENGTH, FEATURE_NUMBER], name="x_1")
            self.trace_lengths = tf.placeholder(tf.int32, [None], name="tl")

            lstm_cell = tf.contrib.rnn.LSTMCell(num_units=H_SIZE, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-1.0, 1.0))

            # single_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=DROPOUT_KEEP_PROB,
            #                                             output_keep_prob=DROPOUT_KEEP_PROB)

            self.cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * RNN_LAYER, state_is_tuple=True)

            self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                inputs=self.rnn_input, cell=self.cell, sequence_length=self.trace_lengths, dtype=tf.float32,
                scope=rnn_type + '_rnn')

            # state_in = single_cell.zero_state(BATCH_SIZE, tf.float32)
            # rnn_output, rnn_state = tf.contrib.rnn.static_rnn(inputs=rnn_input, cell=cell, dtype=tf.float32, scope=rnn_type + '_rnn')

            # tf.contrib.rnn.BasicLSTMCell()  # LSTM with rectifier, don't need dropout wrapper?

            # [batch_size, max_time, cell.output_size]
            self.outputs = tf.stack(self.rnn_output)

            # Hack to build the indexing and retrieve the right output.
            self.batch_size = tf.shape(self.outputs)[0]
            # Start indices for each sample
            self.index = tf.range(0, self.batch_size) * MAX_TRACE_LENGTH + (self.trace_lengths - 1)
            # Indexing
            self.rnn_last = tf.gather(tf.reshape(self.outputs, [-1, H_SIZE]), self.index)

        num_layer_1 = H_SIZE
        num_layer_2 = 1000
        num_layer_3 = 1

        with tf.name_scope("Dense_Layer_first"):
            self.W1 = tf.get_variable('w1_xaiver', [num_layer_1, num_layer_2],
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
            self.y1 = tf.matmul(self.rnn_last, self.W1) + self.b1
            self.activation1 = tf.nn.relu(self.y1, name='activation')

        with tf.name_scope("Dense_Layer_second"):
            self.W2 = tf.get_variable('w2_xaiver', [num_layer_2, num_layer_3],
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b2 = tf.Variable(tf.zeros([num_layer_3]), name="b_1")
            self.read_out = tf.matmul(self.activation1, self.W2) + self.b2

        self.y = tf.placeholder("float", [None, num_layer_3])

        with tf.name_scope("cost"):
            self.readout_action = self.read_out
            self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
        tf.summary.histogram('cost', self.cost)

        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


def handle_trace_length(state_trace_length):
    """
    transform format of trace length
    :return:
    """
    trace_length_record = []
    for length in state_trace_length:
        for sub_length in range(0, length):
            trace_length_record.append(sub_length + 1)
    return trace_length_record


def get_training_batch(s_t0, state_input, reward, train_number, train_len, state_trace_length):
    """
    combine training data to a batch
    :return:
    """
    batch_return = []
    current_batch_length = 0
    while current_batch_length < BATCH_SIZE:
        s_t1 = state_input[train_number]
        if len(s_t1) < 10 or len(s_t0) < 10:
            raise ValueError("wrong length of s")
            # train_number += 1
            # continue
        s_length_t1 = state_trace_length[train_number]
        s_length_t0 = state_trace_length[train_number - 1]
        if s_length_t1 > 10:  # if trace length is too long
            s_length_t1 = 10
        if s_length_t0 > 10:  # if trace length is too long
            s_length_t0 = 10
        try:
            s_reward_t0 = reward[train_number -1]
            s_reward_t1 = reward[train_number]
        except IndexError:
            raise IndexError("s_reward wrong with index")
        train_number += 1
        if train_number + 1 == train_len:
            trace_length_index_t0 = s_length_t0 - 1
            trace_length_index_t1 = s_length_t1 - 1
            r_t0 = np.asarray([s_reward_t0[trace_length_index_t0]])
            r_t1 = np.asarray([s_reward_t1[trace_length_index_t1]])
            batch_return.append((s_t0, s_t1, r_t0, s_length_t0, s_length_t1, 0))
            batch_return.append((s_t1, s_t1, r_t1, s_length_t1, s_length_t1, 1))
            s_t0 = s_t1
            break
        trace_length_index_t0 = s_length_t0 - 1
        r_t0 = np.asarray([s_reward_t0[trace_length_index_t0]])
        if r_t0 != [float(0)]:
            print str(r_t0)
        batch_return.append((s_t0, s_t1, r_t0, s_length_t0, s_length_t1, 0))
        current_batch_length += 1
        s_t0 = s_t1

    return batch_return, train_number, s_t0


def train_network(sess, model, print_parameters=False):
    game_number = 0
    global_counter = 0
    converge_flag = False

    # loading network
    saver = tf.train.Saver()
    merge = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    sess.run(tf.global_variables_initializer())

    while True:
        if converge_flag:
            break
        elif game_number >= number_of_total_game * ITERATE_NUM:
            break
        else:
            converge_flag = True
        for dir_game in DIR_GAMES_ALL:
            game_number += 1
            game_files = os.listdir(DATA_STORE + "/" + dir_game)
            for filename in game_files:
                if "reward" in filename:
                    reward_name = filename
                elif "input" in filename:
                    state_input_name = filename
                elif "output" in filename:
                    state_output_name = filename
                elif "trace" in filename:
                    state_trace_length_name = filename

            reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + reward_name)
            try:
                reward = reward['hybrid_reward']
            except:
                print "\n" + dir_game
                raise ValueError("reward wrong")
            reward_count = sum(reward)
            state_input = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_input_name)
            state_input = (state_input['hybrid_input_state'])
            state_trace_length = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_trace_length_name)
            state_trace_length = (state_trace_length['hybrid_trace_length'])[0]
            state_trace_length = handle_trace_length(state_trace_length)

            print ("\n load file" + str(dir_game) + " success")
            print ("reward number" + str(reward_count))
            if len(state_input) != len(reward) or len(state_trace_length) != len(reward):
                raise Exception('state length does not equal to reward length')

            train_len = len(state_input)
            train_number = 0
            s_t0 = state_input[train_number]
            train_number += 1

            while True:
                # try:
                batch, train_number, s_tl = get_training_batch(s_t0, state_input, reward, train_number,
                                                               train_len, state_trace_length)

                # get the batch variables
                s_t0_batch = [d[0] for d in batch]
                s_t1_batch = [d[1] for d in batch]
                r_t_batch = [d[2] for d in batch]
                trace_t0_batch = [d[3] for d in batch]
                trace_t1_batch = [d[4] for d in batch]
                y_batch = []

                # readout_t1_batch = model.read_out.eval(
                #     feed_dict={model.trace_lengths: trace_t1_batch, model.rnn_input: s_t1_batch})  # get value of s

                [outputs_t1, rnn_last, readout_t1_batch] = sess.run([model.outputs, model.rnn_last, model.read_out],
                                                                    feed_dict={model.trace_lengths: trace_t1_batch,
                                                                               model.rnn_input: s_t1_batch})

                for i in range(0, len(batch)):
                    terminal = batch[i][5]
                    # if r_t_batch[i][-1] != float(0):
                    #     print r_t_batch[i][-1]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append([float(r_t_batch[i][-1])])
                        break
                    else:
                        y_batch.append([r_t_batch[i][-1] + GAMMA * ((readout_t1_batch[i]).tolist())[0]])

                # perform gradient step
                y_batch = np.asarray(y_batch)
                [index, cost_out, summary_train, _] = sess.run(
                    [model.index, model.cost, merge, model.train_step],
                    feed_dict={model.y: y_batch,
                               model.trace_lengths: trace_t0_batch,
                               model.rnn_input: s_t0_batch})

                if cost_out > 0.0001:
                    converge_flag = False
                global_counter += 1
                train_writer.add_summary(summary_train, global_step=global_counter)
                s_t0 = s_tl

                # print info
                if terminal or ((train_number - 1) / BATCH_SIZE) % 5 == 1:
                    print ("TIMESTEP:", train_number, "Game:", game_number)
                    print ("cost of the network is" + str(cost_out))

                if terminal:
                    # save progress after a game
                    saver.save(sess, SAVED_NETWORK + '/' + SPORT + '-game-', global_step=game_number)

                    break


def train_start():
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.isdir(SAVED_NETWORK):
        os.mkdir(SAVED_NETWORK)

    sess = tf.InteractiveSession()
    nn = td_prediction_lstm()
    train_network(sess, nn)


if __name__ == '__main__':
    train_start()
