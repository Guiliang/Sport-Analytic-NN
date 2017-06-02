import random

import scipy.io as sio
import tensorflow as tf
import os
import traceback
import numpy as np
import math

"""
train the home team and away team together, use a feature to represent it.
"""
number_of_total_game = 446
feature_num = 26
FEATURE_TYPE = 5
model_train_continue = True
ITERATE_NUM = 25
# ITERATE_NUM = "2Converge"
REWARD_TYPE = "NEG_REWARD_GAMMA1_V3"
Home_model_or_away_model = "Away"
TRAIN_or_TEST = ""
Random_or_Sequenced = "Sequenced"

if Home_model_or_away_model == "Home":
    Home_model = True
elif Home_model_or_away_model == "Away":
    Home_model = False
else:
    raise ValueError("Home_model_or_away_model setting wrong")

if TRAIN_or_TEST == "TRAIN":
    Train = True
elif TRAIN_or_TEST == "":
    Train = False
else:
    raise ValueError("TRAIN_or_TEST setting wrong")

if Random_or_Sequenced == "Random":
    Random_select = True
elif Random_or_Sequenced == "Sequenced":
    Random_select = False
else:
    raise ValueError("Random_or_Sequenced setting wrong")

GAMMA = 1  # decay rate of past observations
BATCH_SIZE = 16  # size of mini-batch, the size of mini-batch could be tricky, the larger mini-batch, the easier will it be converge, but if our training data is not comprehensive enough and stochastic gradients is not applied, model may converge to other things
SPORT = "NHL"
if Train:
    DATA_STORE = "/cs/oschulte/Galen/Hockey-data/Hockey-Training-All-feature" + str(
        FEATURE_TYPE) + "-scale-neg_reward_Train"
else:
    DATA_STORE = "/cs/oschulte/Galen/Hockey-data/Hockey-Training-All-feature" + str(
        FEATURE_TYPE) + "-scale-neg_reward"

DIR_GAMES_ALL = os.listdir(DATA_STORE)
LOG_DIR = "/cs/oschulte/Galen/models/log_NN/mc_separate_log_" + str(Home_model_or_away_model) + "_train_feature" + str(FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "-" + str(REWARD_TYPE) + TRAIN_or_TEST + "-" + Random_or_Sequenced
SAVED_NETWORK = "/cs/oschulte/Galen/models/saved_NN/mc_separate_saved_" + str(Home_model_or_away_model) + "_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "-" + str(REWARD_TYPE) + TRAIN_or_TEST + "-" + Random_or_Sequenced
FORWARD_REWARD_MODE = False


class mc_prediction_simple(object):
    def __init__(self):
        """
        define the neural network
        :return: network output
        """
        # network weights
        # with tf.name_scope("Dense_Layer_first"):
        #     x = tf.placeholder(tf.float32, [None, feature_num], name="x")
        #     with tf.name_scope('weights'):
        #         W1 = tf.Variable(tf.zeros([feature_num, 1000]), name="W")
        #     with tf.name_scope('biases'):
        #         b1 = tf.Variable(tf.zeros([1000]), name="b")
        #     with tf.name_scope('Wx_plus_b'):
        #         y1 = tf.matmul(x, W1) + b1
        #     activations = tf.nn.relu(y1, name='activation')
        #     tf.summary.histogram('activations', activations)
        #
        # with tf.name_scope("Dense_Layer_second"):
        #     with tf.name_scope('weights'):
        #         W2 = tf.Variable(tf.zeros([1000, 1]), name="W")
        #     with tf.name_scope('biases'):
        #         b2 = tf.Variable(tf.zeros([1]), name="b")
        #     with tf.name_scope('Wx_plus_b'):
        #         read_out = tf.matmul(activations, W2) + b2

        # 7 is the num of units is layer 1
        # 1000 is the num of units in layer 2
        # 1 is the num of unit in layer 3

        num_layer_1 = feature_num
        num_layer_2 = 1000
        num_layer_3 = 1
        max_sigmoid_1 = -4 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
        min_sigmoid_1 = 4 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
        var_sigmoid_1 = float(1) / (num_layer_1 + num_layer_2)
        max_sigmoid_2 = -4 * math.sqrt(float(6) / (num_layer_2 + num_layer_3))
        min_sigmoid_2 = 4 * math.sqrt(float(6) / (num_layer_2 + num_layer_3))
        var_sigmoid_2 = float(1) / (num_layer_2 + num_layer_3)

        with tf.name_scope("Dense_Layer_first"):
            self.x = tf.placeholder(tf.float32, [None, num_layer_1], name="x_1")
            with tf.name_scope("Weight_1"):
                self.W1 = tf.Variable(
                    tf.random_uniform([num_layer_1, num_layer_2], minval=min_sigmoid_1, maxval=max_sigmoid_1),
                    name="W_1")
            with tf.name_scope("Biases_1"):
                self.b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
            with tf.name_scope("Output_1"):
                self.y1 = tf.matmul(self.x, self.W1) + self.b1
            with tf.name_scope("Activation_1"):
                self.activations = tf.nn.sigmoid(self.y1, name='activation')
                tf.summary.histogram('activation_1', self.activations)

        # to debug the network
        self.W1_print = tf.Print(self.W1, [self.W1], message="W1 is:", summarize=40)
        self.y1_print = tf.Print(self.y1, [self.y1], message="y1 is:", summarize=40)
        self.b1_print = tf.Print(self.b1, [self.b1], message="b1 is:", summarize=40)

        with tf.name_scope("Dense_Layer_second"):
            with tf.name_scope("Weight_2"):
                self.W2 = tf.Variable(
                    tf.random_uniform([num_layer_2, num_layer_3], minval=min_sigmoid_2, maxval=max_sigmoid_2),
                    name="W_2")
            with tf.name_scope("Biases_1"):
                self.b2 = tf.Variable(tf.zeros([num_layer_3]), name="b_2")
            with tf.name_scope("Output_2"):
                self.read_out = tf.matmul(self.activations, self.W2) + self.b2
                tf.summary.histogram('output_2', self.activations)

        # to debug the network
        self.W2_print = tf.Print(self.W2, [self.W2], message="W2 is:", summarize=40)
        self.y2_print = tf.Print(self.read_out, [self.read_out], message="y2 is:", summarize=40)
        self.b2_print = tf.Print(self.b2, [self.b2], message="b2 is:", summarize=40)

        # define the cost function
        self.y = tf.placeholder("float", [None])

        with tf.name_scope("cost"):
            self.readout_action = tf.reduce_sum(self.read_out,
                                                reduction_indices=1)  # Computes the sum of elements across dimensions of a tensor.
            self.diff_v = tf.reduce_mean(tf.abs(self.y - self.readout_action))
            self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))  # square means
        tf.summary.histogram('cost', self.cost)

        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
            # self.train_step = tf.train.GradientDescentOptimizer(1e-6).minimize(self.cost)
            # train_step = tf.train.AdadeltaOptimizer().minimize(cost)


class mc_prediction_simple_V2(object):
    def __init__(self):
        """
        define the neural network
        :return: network output
        """

        # 7 is the num of units is layer 1
        # 1000 is the num of units in layer 2
        # 1 is the num of unit in layer 3

        num_layer_1 = feature_num
        num_layer_2 = 1000
        num_layer_3 = 1000
        num_layer_4 = 1

        max_sigmoid_1 = -1 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
        min_sigmoid_1 = 1 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
        max_sigmoid_2 = -1 * math.sqrt(float(6) / (num_layer_2 + num_layer_3))
        min_sigmoid_2 = 1 * math.sqrt(float(6) / (num_layer_2 + num_layer_3))
        max_sigmoid_3 = -1 * math.sqrt(float(6) / (num_layer_3 + num_layer_4))
        min_sigmoid_3 = 1 * math.sqrt(float(6) / (num_layer_3 + num_layer_4))

        with tf.name_scope("Dense_Layer_first"):
            self.x = tf.placeholder(tf.float32, [None, num_layer_1], name="x_1")
            with tf.name_scope("Weight_1"):
                self.W1 = tf.Variable(
                    tf.random_uniform([num_layer_1, num_layer_2], minval=min_sigmoid_1, maxval=max_sigmoid_1),
                    name="W_1")
            with tf.name_scope("Biases_1"):
                self.b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
            with tf.name_scope("Output_1"):
                self.y1 = tf.matmul(self.x, self.W1) + self.b1
            with tf.name_scope("Activation_1"):
                self.activations1 = tf.nn.relu(self.y1, name='activation')

        with tf.name_scope("Dense_Layer_second"):
            with tf.name_scope("Weight_2"):
                self.W2 = tf.Variable(
                    tf.random_uniform([num_layer_2, num_layer_3], minval=min_sigmoid_2, maxval=max_sigmoid_2),
                    name="W_2")
            with tf.name_scope("Biases_2"):
                self.b2 = tf.Variable(tf.zeros([num_layer_3]), name="b_2")
            with tf.name_scope("Output_2"):
                self.y2 = tf.matmul(self.activations1, self.W2) + self.b2
            with tf.name_scope("Activation_2"):
                self.activations2 = tf.nn.relu(self.y2, name='activation')

        with tf.name_scope("Dense_Layer_third"):
            with tf.name_scope("Weight_3"):
                self.W3 = tf.Variable(
                    tf.random_uniform([num_layer_3, num_layer_4], minval=min_sigmoid_3, maxval=max_sigmoid_3),
                    name="W_3")
            with tf.name_scope("Biases_3"):
                self.b3 = tf.Variable(tf.zeros([num_layer_4]), name="b_3")
            with tf.name_scope("Output_3"):
                self.read_out = tf.matmul(self.activations2, self.W3) + self.b3
                # tf.summary.histogram('output_2', self.activations)

        # define the cost function
        self.y = tf.placeholder("float", [None])

        with tf.name_scope("cost"):
            self.readout_action = tf.reduce_sum(self.read_out,
                                                reduction_indices=1)  # Computes the sum of elements across dimensions of a tensor.
            self.diff_v = tf.reduce_mean(tf.abs(self.y - self.readout_action))
            self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))  # square means
        tf.summary.histogram('cost', self.cost)

        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
            # self.train_step = tf.train.GradientDescentOptimizer(1e-6).minimize(self.cost)
            # train_step = tf.train.AdadeltaOptimizer().minimize(cost)


class mc_prediction_simple_V3(object):
    def __init__(self):
        """
        define the neural network
        :return: network output
        """

        # 7 is the num of units is layer 1
        # 1000 is the num of units in layer 2
        # 1 is the num of unit in layer 3

        num_layer_1 = feature_num
        num_layer_2 = 1000
        num_layer_3 = 1000
        num_layer_4 = 1000
        num_layer_5 = 1

        max_sigmoid_1 = -1 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
        min_sigmoid_1 = 1 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
        max_sigmoid_2 = -1 * math.sqrt(float(6) / (num_layer_2 + num_layer_3))
        min_sigmoid_2 = 1 * math.sqrt(float(6) / (num_layer_2 + num_layer_3))
        max_sigmoid_3 = -1 * math.sqrt(float(6) / (num_layer_3 + num_layer_4))
        min_sigmoid_3 = 1 * math.sqrt(float(6) / (num_layer_3 + num_layer_4))
        max_sigmoid_4 = -1 * math.sqrt(float(6) / (num_layer_4 + num_layer_5))
        min_sigmoid_4 = 1 * math.sqrt(float(6) / (num_layer_4 + num_layer_5))

        with tf.name_scope("Dense_Layer_first"):
            self.x = tf.placeholder(tf.float32, [None, num_layer_1], name="x_1")
            with tf.name_scope("Weight_1"):
                self.W1 = tf.Variable(
                    tf.random_uniform([num_layer_1, num_layer_2], minval=min_sigmoid_1, maxval=max_sigmoid_1),
                    name="W_1")
            with tf.name_scope("Biases_1"):
                self.b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
            with tf.name_scope("Output_1"):
                self.y1 = tf.matmul(self.x, self.W1) + self.b1
            with tf.name_scope("Activation_1"):
                self.activations1 = tf.nn.tanh(self.y1, name='activation1')

        with tf.name_scope("Dense_Layer_second"):
            with tf.name_scope("Weight_2"):
                self.W2 = tf.Variable(
                    tf.random_uniform([num_layer_2, num_layer_3], minval=min_sigmoid_2, maxval=max_sigmoid_2),
                    name="W_2")
            with tf.name_scope("Biases_2"):
                self.b2 = tf.Variable(tf.zeros([num_layer_3]), name="b_2")
            with tf.name_scope("Output_2"):
                self.y2 = tf.matmul(self.activations1, self.W2) + self.b2
            with tf.name_scope("Activation_2"):
                self.activations2 = tf.nn.tanh(self.y2, name='activation2')

        with tf.name_scope("Dense_Layer_third"):
            with tf.name_scope("Weight_3"):
                self.W3 = tf.Variable(
                    tf.random_uniform([num_layer_3, num_layer_4], minval=min_sigmoid_3, maxval=max_sigmoid_3),
                    name="W_3")
            with tf.name_scope("Biases_3"):
                self.b3 = tf.Variable(tf.zeros([num_layer_4]), name="b_3")
            with tf.name_scope("Output_3"):
                self.y3 = tf.matmul(self.activations2, self.W3) + self.b3
            with tf.name_scope("Activation_3"):
                self.activations3 = tf.nn.tanh(self.y3, name='activation3')

        with tf.name_scope("Dense_Layer_fourth"):
            with tf.name_scope("Weight_4"):
                self.W4 = tf.Variable(
                    tf.random_uniform([num_layer_4, num_layer_5], minval=min_sigmoid_4, maxval=max_sigmoid_4),
                    name="W_4")
            with tf.name_scope("Biases_4"):
                self.b4 = tf.Variable(tf.zeros([num_layer_5]), name="b_4")
            with tf.name_scope("Output_4"):
                self.read_out = tf.matmul(self.activations3, self.W4) + self.b4

        # define the cost function
        self.y = tf.placeholder("float", [None])

        with tf.name_scope("cost"):
            self.readout_action = tf.reduce_sum(self.read_out,
                                                reduction_indices=1)  # Computes the sum of elements across dimensions of a tensor.
            self.diff_v = tf.reduce_mean(tf.abs(self.y - self.readout_action))
            self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))  # square means
        tf.summary.histogram('cost', self.cost)

        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
            # self.train_step = tf.train.GradientDescentOptimizer(1e-6).minimize(self.cost)
            # train_step = tf.train.AdadeltaOptimizer().minimize(cost)


def reward_append(reward, reward_record):
    for reward_index in range(0, len(reward_record)):
        reward_value = (reward_record[reward_index])
        if reward_value != "Null":
            reward_value += reward
            (reward_record[reward_index]) = reward_value

    return reward_record


def build_mc_training_batch(state, target_reward, train_number):
    training_batch = []
    if train_number + BATCH_SIZE < len(state):
        state_batch = state[train_number:train_number + BATCH_SIZE]
        reward_batch = target_reward[train_number:train_number + BATCH_SIZE]
        train_number += BATCH_SIZE
        training_batch.append((state_batch, reward_batch, 0))
    else:
        state_batch = state[train_number:len(state)]
        reward_batch = target_reward[train_number:len(state)]
        train_number += BATCH_SIZE
        training_batch.append((state_batch, reward_batch, 1))

    return training_batch, train_number


def train_network(sess, model):
    """
    train the network
    :param print_parameters:
    :return:
    """
    game_number = 0
    global_counter = 0
    converge_flag = False

    # loading network
    saver = tf.train.Saver()
    merge = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    sess.run(tf.global_variables_initializer())
    print SAVED_NETWORK
    if model_train_continue:
        checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORK)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    while True:
        print str(number_of_total_game * ITERATE_NUM)
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
                if filename.startswith("reward"):
                    reward_name = filename
                elif filename.startswith("state"):
                    state_name = filename
                elif filename.startswith("time"):
                    time_remain_name = filename

            reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + reward_name)
            try:
                reward = (reward['reward'][0]).tolist()
            except:
                print "\n" + dir_game
                continue
            reward_count = sum(reward)

            state = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_name)
            state = state['state']
            time_remain = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + time_remain_name)
            time_remain = (time_remain['time_remain'])[0]

            print ("\n load file" + str(dir_game) + " success")
            print ("reward number" + str(reward_count))
            if len(state) != len(reward):
                raise Exception('state length does not equal to reward length')

            train_number = 0

            if Home_model:
                target_reward_home_record = []
            else:
                target_reward_away_record = []

            for reward_index in range(0, len(reward)):
                if Home_model:
                    target_reward_home_record.append(0)
                    if reward[reward_index] == 1:
                        reward_append(1, target_reward_home_record)
                else:
                    target_reward_away_record.append(0)
                    if reward[reward_index] == -1:
                        reward_append(-1, target_reward_away_record)

            if Home_model:
                target_reward_home_record = [float(a) / float(b) for a, b in zip(target_reward_home_record, time_remain)]
            else:
                target_reward_away_record = [float(a) / float(b) for a, b in zip(target_reward_away_record, time_remain)]

            while True:
                if Home_model:
                    training_batch_home, train_number = build_mc_training_batch(state, target_reward_home_record,
                                                                                train_number)
                else:
                    training_batch_away, train_number = build_mc_training_batch(state, target_reward_away_record,
                                                                                train_number)

                if Home_model:
                    s_t_home_batch = [d[0] for d in training_batch_home]
                    r_t_home_batch = [d[1] for d in training_batch_home]
                    terminal = (training_batch_home[-1])[-1]
                    s_t_batch = s_t_home_batch
                    r_t_batch = r_t_home_batch
                elif not Home_model:
                    s_t_away_batch = [d[0] for d in training_batch_away]
                    r_t_away_batch = [d[1] for d in training_batch_away]
                    terminal = (training_batch_away[-1])[-1]
                    s_t_batch = s_t_away_batch
                    r_t_batch = r_t_away_batch
                else:
                    raise ValueError("Home or away model can't match")

                # perform gradient step
                [diff_v, cost_out, summary_train, _] = sess.run([model.diff_v, model.cost, merge, model.train_step],
                                                                feed_dict={model.y: r_t_batch[0],
                                                                           model.x: s_t_batch[0]})
                if diff_v > 0.01:
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

    train_writer.close()


def train_start():
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.isdir(SAVED_NETWORK):
        os.mkdir(SAVED_NETWORK)

    sess = tf.InteractiveSession()
    nn = mc_prediction_simple_V3()
    train_network(sess, nn)


if __name__ == '__main__':
    train_start()
