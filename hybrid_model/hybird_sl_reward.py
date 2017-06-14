import tensorflow as tf
import math
import os
import scipy.io as sio
import traceback
import numpy as np

MAX_TRACE_LENGTH = 10
FEATURE_NUMBER = 24
BATCH_SIZE = 16
GAMMA = 1
H_SIZE = 512
DROPOUT_KEEP_PROB = 0.8
RNN_LAYER = 2
USE_HIDDEN_STATE = False
FEATURE_TYPE = 7

SPORT = "NHL"
ITERATE_NUM = "2Converge"
REWARD_TYPE = "NEG_REWARD_GAMMA1_V3"
DATA_STORE = "/cs/oschulte/Galen/Hockey-data/Hybrid-RNN-Hockey-Training-All-feature7-scale-neg_reward-length-dynamic"
LOG_DIR = "/cs/oschulte/Galen/models/hybrid_sl_log_NN/log_train_feature" + str(FEATURE_TYPE) + "_batch" + str(BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM)
SAVED_NETWORK = "/cs/oschulte/Galen/models/hybrid_sl_saved_NN/saved_networks_feature" + str(FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM)
DIR_GAMES_ALL = os.listdir(DATA_STORE)


class create_network_hybrid_sl:
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

            single_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=DROPOUT_KEEP_PROB,
                                                        output_keep_prob=DROPOUT_KEEP_PROB)

            self.cell = tf.contrib.rnn.MultiRNNCell([single_cell] * RNN_LAYER, state_is_tuple=True)

            self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                inputs=self.rnn_input, cell=self.cell, sequence_length=self.trace_lengths, dtype=tf.float32,
                scope=rnn_type + '_rnn')

            # state_in = single_cell.zero_state(BATCH_SIZE, tf.float32)
            # rnn_output, rnn_state = tf.contrib.rnn.static_rnn(inputs=rnn_input, cell=cell, dtype=tf.float32, scope=rnn_type + '_rnn')

            # tf.contrib.rnn.BasicLSTMCell()  # LSTM with rectifier, don't need dropout wrapper?

            # [batch_size, max_time, cell.output_size]
            outputs = tf.stack(self.rnn_output)

            # Hack to build the indexing and retrieve the right output.
            self.batch_size = tf.shape(outputs)[0]
            # Start indices for each sample
            self.index = tf.range(0, self.batch_size) * MAX_TRACE_LENGTH + (self.trace_lengths - 1)
            # Indexing
            self.rnn_last = tf.gather(tf.reshape(outputs, [-1, H_SIZE]), self.index)

        num_layer_1 = H_SIZE
        num_layer_2 = 1  # feature + reward
        max_sigmoid_1 = math.sqrt(float(6) / (num_layer_1 + num_layer_2))
        min_sigmoid_1 = -math.sqrt(float(6) / (num_layer_1 + num_layer_2))

        with tf.name_scope("Dense_Layer_first"):
            self.w1 = tf.Variable(
                tf.random_uniform([num_layer_1, num_layer_2], minval=min_sigmoid_1, maxval=max_sigmoid_1),
                name="W_1")
            self.b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
            self.y1 = tf.matmul(self.rnn_last, self.w1) + self.b1
            self.read_out = tf.nn.tanh(self.y1, name='activation')

        self.y = tf.placeholder("float", [None, num_layer_2])

        with tf.name_scope("cost"):
            self.readout_action = self.read_out
            self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
        tf.summary.histogram('cost', self.cost)

        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


def get_training_batch(state_input, state_output, reward, train_number, train_len, state_trace_length):
    """
    combine training data to a batch
    :return:
    """
    batch_return = []
    current_batch_length = 0
    while current_batch_length < BATCH_SIZE:
        s_input = state_input[train_number]
        if len(s_input) < 10:
            print "jump to next state as len(state) is not 10"
            train_number += 1
            continue
        s_output = state_output[train_number]
        s_length = state_trace_length[train_number]
        if s_length > 10:  # if trace length is too long
            s_length = 10
        try:
            s_reward = reward[train_number]
        except IndexError:
            raise IndexError("s_reward wrong with index")
        train_number += 1
        if train_number + 1 == train_len:
            trace_length_index = s_length - 1
            batch_return.append((s_input, s_output, np.asarray([s_reward[trace_length_index]]), s_length, 1))
            break
        trace_length_index = s_length - 1
        batch_return.append((s_input, s_output, np.asarray([s_reward[trace_length_index]]), s_length, 0))
        current_batch_length += 1

    return batch_return, train_number


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
            state_output = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_output_name)
            state_output = state_output['hybrid_output_state']
            state_trace_length = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_trace_length_name)
            state_trace_length = (state_trace_length['hybrid_trace_length'])[0]

            print ("\n load file" + str(dir_game) + " success")
            print ("reward number" + str(reward_count))
            if len(state_output) != len(reward) or len(state_input) != len(reward):
                raise Exception('state length does not equal to reward length')

            train_len = len(state_input)
            train_number = 0

            t_batch_pre = 0
            while True:
                try:
                    batch, train_number = get_training_batch(state_input, state_output, reward, train_number,
                                                             train_len, state_trace_length)

                    # get the batch variables
                    s_input_batch = [d[0] for d in batch]
                    # s_output_batch = [d[1] for d in batch]
                    r_batch = [d[2] for d in batch]
                    t_batch = [d[3] for d in batch]
                    terminal = ([d[4] for d in batch])[-1]

                    y_batch = []
                    readout_t1_batch = model.read_out.eval(feed_dict={model.rnn_input: s_t1_batch})

                    # perform gradient step

                    [index, cost_out, summary_train, _] = sess.run([model.index, model.cost, merge, model.train_step],
                                                                   feed_dict={model.y: y_batch,
                                                                              model.trace_lengths: t_batch,
                                                                              model.rnn_input: s_input_batch})
                except:
                    print "\n" + dir_game
                    raise ValueError("Train network wrong!")
                    traceback.print_exc()

                if cost_out > 0.0001:
                    converge_flag = False
                global_counter += 1
                train_writer.add_summary(summary_train, global_step=global_counter)

                # print info
                if terminal or ((train_number - 1) / BATCH_SIZE) % 5 == 1:
                    print ("TIMESTEP:", train_number, "Game:", game_number)
                    print ("cost of the network is" + str(cost_out))

                if terminal:
                    # save progress after a game
                    saver.save(sess, SAVED_NETWORK + '/' + SPORT + '-game-', global_step=game_number)

                    break

                t_batch_pre = t_batch


def train_start():
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.isdir(SAVED_NETWORK):
        os.mkdir(SAVED_NETWORK)

    sess = tf.InteractiveSession()
    nn = create_network_hybrid_sl()
    train_network(sess, nn)


if __name__ == '__main__':
    train_start()
