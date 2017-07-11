import csv

import tensorflow as tf
import math
import os
import scipy.io as sio
import traceback
import numpy as np

MODEL_TYPE = "v3"
MAX_TRACE_LENGTH = 10
FEATURE_NUMBER = 25
BATCH_SIZE = 8
GAMMA = 1
H_SIZE = 512
# DROPOUT_KEEP_PROB = 0.8
# RNN_LAYER = 2
USE_HIDDEN_STATE = False
model_train_continue = True
SCALE = True
FEATURE_TYPE = 5
ITERATE_NUM = 50
learning_rate = 1e-4
SPORT = "NHL"
REWARD_TYPE = "NEG_REWARD_GAMMA1_V3"
save_mother_dir = "/local-scratch"

if SCALE:
    LOG_DIR = save_mother_dir + "/oschulte/Galen/models/hybrid_sl_log_NN/Scale-cut_together_log_train_feature" + str(
        FEATURE_TYPE) + "_batch" + str(
        BATCH_SIZE) + "_iterate" + str(
        ITERATE_NUM) + "_lr" + str(
        learning_rate) + "_" + str(MODEL_TYPE)
    SAVED_NETWORK = save_mother_dir + "/oschulte/Galen/models/hybrid_sl_saved_NN/Scale-cut_together_saved_networks_feature" + str(
        FEATURE_TYPE) + "_batch" + str(
        BATCH_SIZE) + "_iterate" + str(
        ITERATE_NUM) + "_lr" + str(
        learning_rate) + "_" + str(MODEL_TYPE)
    DATA_STORE = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature" + str(
        FEATURE_TYPE) + "-scale-neg_reward_length-dynamic"
else:
    LOG_DIR = save_mother_dir + "/oschulte/Galen/models/hybrid_sl_log_NN/cut_together_log_train_feature" + str(
        FEATURE_TYPE) + "_batch" + str(
        BATCH_SIZE) + "_iterate" + str(
        ITERATE_NUM) + "_lr" + str(
        learning_rate) + "_" + str(MODEL_TYPE)
    SAVED_NETWORK = save_mother_dir + "/oschulte/Galen/models/hybrid_sl_saved_NN/cut_together_saved_networks_feature" + str(
        FEATURE_TYPE) + "_batch" + str(
        BATCH_SIZE) + "_iterate" + str(
        ITERATE_NUM) + "_lr" + str(
        learning_rate) + "_" + str(MODEL_TYPE)
    DATA_STORE = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature" + str(
        FEATURE_TYPE) + "-neg_reward_length-dynamic"

DIR_GAMES_ALL = os.listdir(DATA_STORE)
number_of_total_game = len(DIR_GAMES_ALL)


# class td_prediction_lstm:
#     def __init__(self, rnn_type='bp_last_step'):
#         """
#         define the neural network
#         :return: network output
#         """
#         with tf.name_scope("LSTM_layer"):
#             self.rnn_input = tf.placeholder(tf.float32, [None, MAX_TRACE_LENGTH, FEATURE_NUMBER], name="x_1")
#             self.trace_lengths = tf.placeholder(tf.int32, [None], name="tl")
#
#             lstm_cell = tf.contrib.rnn.LSTMCell(num_units=H_SIZE, state_is_tuple=True,
#                                                 initializer=tf.random_uniform_initializer(-1.0, 1.0))
#
#             # single_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=DROPOUT_KEEP_PROB,
#             #                                             output_keep_prob=DROPOUT_KEEP_PROB)
#
#             self.cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * RNN_LAYER, state_is_tuple=True)
#
#             self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
#                 inputs=self.rnn_input, cell=self.cell, sequence_length=self.trace_lengths, dtype=tf.float32,
#                 scope=rnn_type + '_rnn')
#
#             # state_in = single_cell.zero_state(BATCH_SIZE, tf.float32)
#             # rnn_output, rnn_state = tf.contrib.rnn.static_rnn(inputs=rnn_input, cell=cell, dtype=tf.float32, scope=rnn_type + '_rnn')
#
#             # tf.contrib.rnn.BasicLSTMCell()  # LSTM with rectifier, don't need dropout wrapper?
#
#             # [batch_size, max_time, cell.output_size]
#             self.outputs = tf.stack(self.rnn_output)
#
#             # Hack to build the indexing and retrieve the right output.
#             self.batch_size = tf.shape(self.outputs)[0]
#             # Start indices for each sample
#             self.index = tf.range(0, self.batch_size) * MAX_TRACE_LENGTH + (self.trace_lengths - 1)
#             # Indexing
#             self.rnn_last = tf.gather(tf.reshape(self.outputs, [-1, H_SIZE]), self.index)
#
#         num_layer_1 = H_SIZE
#         num_layer_2 = 1000
#         num_layer_3 = 1
#
#         with tf.name_scope("Dense_Layer_first"):
#             self.W1 = tf.get_variable('w1_xaiver', [num_layer_1, num_layer_2],
#                                       initializer=tf.contrib.layers.xavier_initializer())
#             self.b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
#             self.y1 = tf.matmul(self.rnn_last, self.W1) + self.b1
#             self.activation1 = tf.nn.relu(self.y1, name='activation')
#
#         with tf.name_scope("Dense_Layer_second"):
#             self.W2 = tf.get_variable('w2_xaiver', [num_layer_2, num_layer_3],
#                                       initializer=tf.contrib.layers.xavier_initializer())
#             self.b2 = tf.Variable(tf.zeros([num_layer_3]), name="b_1")
#             self.read_out = tf.matmul(self.activation1, self.W2) + self.b2
#
#         self.y = tf.placeholder("float", [None, num_layer_3])
#
#         with tf.name_scope("cost"):
#             self.readout_action = self.read_out
#             self.diff = tf.reduce_mean(tf.abs(self.y - self.readout_action))
#             self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
#         tf.summary.histogram('cost', self.cost)
#
#         with tf.name_scope("train"):
#             self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


# class td_prediction_lstm_v2:
#     def __init__(self, rnn_type='bp_last_step'):
#         """
#         define the neural network
#         :return: network output
#         """
#         with tf.name_scope("LSTM_layer"):
#             self.rnn_input = tf.placeholder(tf.float32, [None, MAX_TRACE_LENGTH, FEATURE_NUMBER], name="x_1")
#             self.trace_lengths = tf.placeholder(tf.int32, [None], name="tl")
#
#             lstm_cell = tf.contrib.rnn.LSTMCell(num_units=H_SIZE, state_is_tuple=True,
#                                                 initializer=tf.random_uniform_initializer(-1.0, 1.0))
#
#             # single_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=DROPOUT_KEEP_PROB,
#             #                                             output_keep_prob=DROPOUT_KEEP_PROB)
#
#             self.cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * RNN_LAYER, state_is_tuple=True)
#
#             self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
#                 inputs=self.rnn_input, cell=self.cell, sequence_length=self.trace_lengths, dtype=tf.float32,
#                 scope=rnn_type + '_rnn')
#
#             # state_in = single_cell.zero_state(BATCH_SIZE, tf.float32)
#             # rnn_output, rnn_state = tf.contrib.rnn.static_rnn(inputs=rnn_input, cell=cell, dtype=tf.float32, scope=rnn_type + '_rnn')
#
#             # tf.contrib.rnn.BasicLSTMCell()  # LSTM with rectifier, don't need dropout wrapper?
#
#             # [batch_size, max_time, cell.output_size]
#             self.outputs = tf.stack(self.rnn_output)
#
#             # Hack to build the indexing and retrieve the right output.
#             self.batch_size = tf.shape(self.outputs)[0]
#             # Start indices for each sample
#             self.index = tf.range(0, self.batch_size) * MAX_TRACE_LENGTH + (self.trace_lengths - 1)
#             # Indexing
#             self.rnn_last = tf.gather(tf.reshape(self.outputs, [-1, H_SIZE]), self.index)
#
#         num_layer_1 = H_SIZE
#         num_layer_2 = 1
#
#         with tf.name_scope("Dense_Layer_first"):
#             self.W1 = tf.get_variable('w1_xaiver', [num_layer_1, num_layer_2],
#                                       initializer=tf.contrib.layers.xavier_initializer())
#             self.b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
#             self.read_out = tf.matmul(self.rnn_last, self.W1) + self.b1
#
#         self.y = tf.placeholder("float", [None, num_layer_2])
#
#         with tf.name_scope("cost"):
#             self.readout_action = self.read_out
#             self.diff = tf.reduce_mean(tf.abs(self.y - self.readout_action))
#             self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
#         tf.summary.histogram('cost', self.cost)
#
#         with tf.name_scope("train"):
#             self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


class td_prediction_lstm_V3:
    def __init__(self, rnn_type='bp_last_step'):
        """
        define the neural network
        :return: network output
        """
        with tf.name_scope("LSTM_layer"):
            self.rnn_input = tf.placeholder(tf.float32, [None, MAX_TRACE_LENGTH, FEATURE_NUMBER], name="x_1")
            self.trace_lengths = tf.placeholder(tf.int32, [None], name="tl")

            self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units=H_SIZE, state_is_tuple=True,
                                                     initializer=tf.random_uniform_initializer(-0.05, 0.05))

            self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                inputs=self.rnn_input, cell=self.lstm_cell, sequence_length=self.trace_lengths, dtype=tf.float32,
                scope=rnn_type + '_rnn')

            # [batch_size, max_time, cell.output_size]
            self.outputs = tf.stack(self.rnn_output)

            # Hack to build the indexing and retrieve the right output.
            self.batch_size = tf.shape(self.outputs)[0]
            # Start indices for each sample
            self.index = tf.range(0, self.batch_size) * MAX_TRACE_LENGTH + (self.trace_lengths - 1)
            # Indexing
            self.rnn_last = tf.gather(tf.reshape(self.outputs, [-1, H_SIZE]), self.index)

        num_layer_1 = H_SIZE
        num_layer_2 = 2

        with tf.name_scope("Dense_Layer_first"):
            self.W1 = tf.get_variable('w1_xaiver', [num_layer_1, num_layer_2],
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
            self.read_out = tf.matmul(self.rnn_last, self.W1) + self.b1
            # self.activation1 = tf.nn.relu(self.y1, name='activation')

        self.y = tf.placeholder("float", [None, num_layer_2])

        with tf.name_scope("cost"):
            self.readout_action = self.read_out
            self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
            self.diff = tf.reduce_mean(tf.abs(self.y - self.readout_action))
        tf.summary.histogram('cost', self.cost)

        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


def write_game_average_csv(data_record):
    try:
        if os.path.exists(LOG_DIR + '/avg_cost_record.csv'):
            with open(LOG_DIR + '/avg_cost_record.csv', 'a') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for record in data_record:
                    writer.writerow(record)
        else:
            with open(LOG_DIR + '/avg_cost_record.csv', 'w') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in data_record:
                    writer.writerow(record)
    except:
        if os.path.exists(LOG_DIR + '/avg_cost_record2.csv'):
            with open(LOG_DIR + '/avg_cost_record.csv', 'a') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for record in data_record:
                    writer.writerow(record)
        else:
            with open(LOG_DIR + '/avg_cost_record2.csv', 'w') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in data_record:
                    writer.writerow(record)

def handle_trace_length(state_trace_length):
    """
    transform format of trace length
    :return:
    """
    trace_length_record = []
    try:
        for length in state_trace_length:
            for sub_length in range(0, int(length)):
                trace_length_record.append(sub_length + 1)
    except:
        print "error"
    return trace_length_record


def get_together_training_batch(s_t0, state_input, reward, train_number, train_len, state_trace_length):
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
            s_reward_t1 = reward[train_number]
            s_reward_t0 = reward[train_number - 1]
        except IndexError:
            raise IndexError("s_reward wrong with index")
        train_number += 1
        if train_number + 1 == train_len:
            trace_length_index_t1 = s_length_t1 - 1
            trace_length_index_t0 = s_length_t0 - 1
            r_t0 = np.asarray([s_reward_t0[trace_length_index_t0]])
            r_t1 = np.asarray([s_reward_t1[trace_length_index_t1]])
            if r_t0 == [float(0)]:
                r_t0_combine = [float(0), float(0)]
                batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 0))

                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0)]
                else:
                    raise ValueError("incorrect r_t1")
                batch_return.append((s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, 1, 0))

            elif r_t0 == [float(-1)]:
                r_t0_combine = [float(0), float(1)]
                batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 0))

                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0)]
                else:
                    raise ValueError("incorrect r_t1")
                batch_return.append((s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, 1, 0))

            elif r_t0 == [float(1)]:
                r_t0_combine = [float(1), float(0)]
                batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 0))

                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0)]
                else:
                    raise ValueError("incorrect r_t1")
                batch_return.append((s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, 1, 0))
            else:
                raise ValueError("r_t0 wrong value")

            s_t0 = s_t1
            break

        trace_length_index_t0 = s_length_t0 - 1  # we want the reward of s_t0, so -2
        r_t0 = np.asarray([s_reward_t0[trace_length_index_t0]])
        if r_t0 != [float(0)]:
            print r_t0
            if r_t0 == [float(-1)]:
                r_t0_combine = [float(0), float(1)]
                batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 1))
            elif r_t0 == [float(1)]:
                r_t0_combine = [float(1), float(0)]
                batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 1))
            else:
                raise ValueError("r_t0 wrong value")
            s_t0 = s_t1
            break
        r_t0_combine = [float(0), float(0)]
        batch_return.append((s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, 0, 0))
        current_batch_length += 1
        s_t0 = s_t1

    return batch_return, train_number, s_t0


def write_csv(csv_name, data_record):
    with open(csv_name, 'w') as csvfile:
        fieldnames = (data_record[0]).keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for record in data_record:
            writer.writerow(record)


def train_network(sess, model, print_parameters=False):
    game_number = 0
    global_counter = 0
    converge_flag = False

    # loading network
    saver = tf.train.Saver()
    merge = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    sess.run(tf.global_variables_initializer())
    if model_train_continue:
        checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORK)
        if checkpoint and checkpoint.model_checkpoint_path:
            check_point_game_number = int((checkpoint.model_checkpoint_path.split("-"))[-1])
            game_number_checkpoint = check_point_game_number % number_of_total_game
            game_number = check_point_game_number
            game_starting_point = 0
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    game_diff_record_all = []

    while True:
        game_diff_record_dict = {}
        iteration_now = game_number / number_of_total_game + 1
        game_diff_record_dict.update({"Iteration": iteration_now})
        if converge_flag:
            break
        elif game_number >= number_of_total_game * ITERATE_NUM:
            break
        else:
            converge_flag = True
        for dir_game in DIR_GAMES_ALL:

            if checkpoint and checkpoint.model_checkpoint_path:
                if model_train_continue:  # go the check point data
                    game_starting_point += 1
                    if game_number_checkpoint + 1 > game_starting_point:
                        continue

            v_diff_record = []
            game_number += 1
            game_cost_record = []
            game_files = os.listdir(DATA_STORE + "/" + dir_game)
            for filename in game_files:
                if "dynamic_rnn_reward" in filename:
                    reward_name = filename
                elif "dynamic_rnn_input" in filename:
                    state_input_name = filename
                elif "trace" in filename:
                    state_trace_length_name = filename

            reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + reward_name)
            try:
                reward = reward['dynamic_rnn_reward']
            except:
                print "\n" + dir_game
                raise ValueError("reward wrong")
            reward_count = sum(reward)
            state_input = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_input_name)
            state_input = (state_input['dynamic_feature_input'])
            # state_output = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_output_name)
            # state_output = state_output['hybrid_output_state']
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
                batch_return, train_number, s_tl = get_together_training_batch(s_t0,
                                                                               state_input,
                                                                               reward,
                                                                               train_number,
                                                                               train_len,
                                                                               state_trace_length)

                # get the batch variables
                s_t0_batch = [d[0] for d in batch_return]
                s_t1_batch = [d[1] for d in batch_return]
                r_t_batch = [d[2] for d in batch_return]
                trace_t0_batch = [d[3] for d in batch_return]
                trace_t1_batch = [d[4] for d in batch_return]
                y_batch = []

                # readout_t1_batch = model.read_out.eval(
                #     feed_dict={model.trace_lengths: trace_t1_batch, model.rnn_input: s_t1_batch})  # get value of s

                [outputs_t1, readout_t1_batch] = sess.run([model.outputs, model.read_out],
                                                          feed_dict={model.trace_lengths: trace_t1_batch,
                                                                     model.rnn_input: s_t1_batch})

                for i in range(0, len(batch_return)):
                    terminal = batch_return[i][5]
                    cut = batch_return[i][6]
                    # if terminal, only equals reward
                    if terminal or cut:
                        y_home = float((r_t_batch[i])[0])
                        y_away = float((r_t_batch[i])[1])
                        y_batch.append([y_home, y_away])
                        break
                    else:
                        y_home = float((r_t_batch[i])[0]) + GAMMA * ((readout_t1_batch[i]).tolist())[0]
                        y_away = float((r_t_batch[i])[1]) + GAMMA * ((readout_t1_batch[i]).tolist())[1]
                        y_batch.append([y_home, y_away])

                # perform gradient step
                y_batch = np.asarray(y_batch)
                [diff, index, cost_out, summary_train, _] = sess.run(
                    [model.diff, model.index, model.cost, merge, model.train_step],
                    feed_dict={model.y: y_batch,
                               model.trace_lengths: trace_t0_batch,
                               model.rnn_input: s_t0_batch})

                v_diff_record.append(diff)

                if cost_out > 0.0001:
                    converge_flag = False
                global_counter += 1
                game_cost_record.append(cost_out)
                train_writer.add_summary(summary_train, global_step=global_counter)
                s_t0 = s_tl

                # print info
                if terminal or ((train_number - 1) / BATCH_SIZE) % 5 == 1:
                    print ("TIMESTEP:", train_number, "Game:", game_number)
                    print ("cost of the network is" + str(cost_out))

                if terminal:
                    # save progress after a game
                    saver.save(sess, SAVED_NETWORK + '/' + SPORT + '-game-', global_step=game_number)
                    v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
                    game_diff_record_dict.update({dir_game: v_diff_record_average})
                    break

                    # break
            cost_per_game_average = sum(game_cost_record) / len(game_cost_record)
            write_game_average_csv([{"iteration": str(game_number / number_of_total_game + 1), "game": game_number,
                                     "cost_per_game_average": cost_per_game_average}])

        game_diff_record_all.append(game_diff_record_dict)
        # break

        # write_csv(SAVE_ITERATION_DIFF_CSV_NAME, game_diff_record_all)


def train_start():
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.isdir(SAVED_NETWORK):
        os.mkdir(SAVED_NETWORK)

    sess = tf.InteractiveSession()
    if MODEL_TYPE == "v3":
        nn = td_prediction_lstm_V3()
    # elif MODEL_TYPE == "v2":
    #     nn = td_prediction_lstm_v2()
    else:
        raise ValueError("MODEL_TYPE error")
    train_network(sess, nn)


if __name__ == '__main__':
    train_start()
