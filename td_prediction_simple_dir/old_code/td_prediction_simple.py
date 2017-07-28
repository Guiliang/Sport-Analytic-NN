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

feature_num = 26
FEATURE_TYPE = 5
model_train_continue = False
ITERATE_NUM = 50
REWARD_TYPE = "NEG_REWARD_GAMMA1"
MODEL_TYPE = "V3"
TRAIN_or_TEST = ""
Random_or_Sequenced = "Sequenced"
GAMMA = 1  # decay rate of past observations
BATCH_SIZE = 16  # size of mini-batch, the size of mini-batch could be tricky, the larger mini-batch, the easier will it be converge, but if our training data is not comprehensive enough and stochastic gradients is not applied, model may converge to other things
SPORT = "NHL"

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
if Train:
    DATA_STORE = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Training-All-feature" + str(
        FEATURE_TYPE) + "-scale-neg_reward_Train"
else:
    DATA_STORE = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Training-All-feature" + str(
        FEATURE_TYPE) + "-scale-neg_reward"

DIR_GAMES_ALL = os.listdir(DATA_STORE)
number_of_total_game = len(DIR_GAMES_ALL)
LOG_DIR = "/cs/oschulte/Galen/models/log_NN/log_entire_" + "_train_feature" + str(
    FEATURE_TYPE) + "_batch" + str(BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "-" + str(REWARD_TYPE) + "_" + MODEL_TYPE + TRAIN_or_TEST + "-" + Random_or_Sequenced
SAVED_NETWORK = "/cs/oschulte/Galen/models/saved_NN/saved_entire_" + "_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "-" + str(REWARD_TYPE) + "_" + MODEL_TYPE + TRAIN_or_TEST + "-" + Random_or_Sequenced
FORWARD_REWARD_MODE = False


class td_prediction_simple(object):
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


class td_prediction_simple_V2(object):
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


class td_prediction_simple_V3(object):
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


class td_prediction_simple_V4(object):
    def __init__(self):
        """
        define the neural network
        :return: network output
        """

        # 7 is the num of units is layer 1
        # 1000 is the num of units in layer 2
        # 1 is the num of unit in layer 3

        num_layer_1 = feature_num
        num_layer_2 = 10000
        num_layer_3 = 10000
        num_layer_4 = 10000
        num_layer_5 = 10000
        num_layer_6 = 1

        max_sigmoid_1 = -1 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
        min_sigmoid_1 = 1 * math.sqrt(float(6) / (num_layer_1 + num_layer_2))
        max_sigmoid_2 = -1 * math.sqrt(float(6) / (num_layer_2 + num_layer_3))
        min_sigmoid_2 = 1 * math.sqrt(float(6) / (num_layer_2 + num_layer_3))
        max_sigmoid_3 = -1 * math.sqrt(float(6) / (num_layer_3 + num_layer_4))
        min_sigmoid_3 = 1 * math.sqrt(float(6) / (num_layer_3 + num_layer_4))
        max_sigmoid_4 = -1 * math.sqrt(float(6) / (num_layer_4 + num_layer_5))
        min_sigmoid_4 = 1 * math.sqrt(float(6) / (num_layer_4 + num_layer_5))
        max_sigmoid_5 = -1 * math.sqrt(float(6) / (num_layer_5 + num_layer_6))
        min_sigmoid_5 = 1 * math.sqrt(float(6) / (num_layer_5 + num_layer_6))

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
                self.y4 = tf.matmul(self.activations3, self.W4) + self.b4
            with tf.name_scope("Activation_4"):
                self.activations4 = tf.nn.tanh(self.y4, name='activation4')

        with tf.name_scope("Dense_Layer_fifth"):
            with tf.name_scope("Weight_5"):
                self.W5 = tf.Variable(
                    tf.random_uniform([num_layer_5, num_layer_6], minval=min_sigmoid_5, maxval=max_sigmoid_5),
                    name="W_5")
            with tf.name_scope("Biases_5"):
                self.b5 = tf.Variable(tf.zeros([num_layer_6]), name="b_5")
            with tf.name_scope("Output_5"):
                self.read_out = tf.matmul(self.activations4, self.W5) + self.b5

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


class td_prediction_simple_V5(object):
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
            self.global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.00001
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                            50000, 0.96, staircase=True)
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost,
                                                                                             global_step=self.global_step)

            # self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
            # self.train_step = tf.train.GradientDescentOptimizer(1e-6).minimize(self.cost)
            # train_step = tf.train.AdadeltaOptimizer().minimize(cost)


class td_prediction_simple_V6(object):
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
            with tf.name_scope("BN_1"):
                self.bn1 = tf.contrib.layers.batch_norm(self.y1,
                                                        center=True, scale=True,
                                                        is_training=True,
                                                        scope='bn1')
            with tf.name_scope("Activation_1"):
                self.activations1 = tf.nn.tanh(self.bn1, name='activation1')

        with tf.name_scope("Dense_Layer_second"):
            with tf.name_scope("Weight_2"):
                self.W2 = tf.Variable(
                    tf.random_uniform([num_layer_2, num_layer_3], minval=min_sigmoid_2, maxval=max_sigmoid_2),
                    name="W_2")
            with tf.name_scope("Biases_2"):
                self.b2 = tf.Variable(tf.zeros([num_layer_3]), name="b_2")
            with tf.name_scope("Output_2"):
                self.y2 = tf.matmul(self.activations1, self.W2) + self.b2
            with tf.name_scope("BN_2"):
                self.bn2 = tf.contrib.layers.batch_norm(self.y2,
                                                        center=True, scale=True,
                                                        is_training=True,
                                                        scope='bn2')
            with tf.name_scope("Activation_2"):
                self.activations2 = tf.nn.tanh(self.bn2, name='activation2')

        with tf.name_scope("Dense_Layer_third"):
            with tf.name_scope("Weight_3"):
                self.W3 = tf.Variable(
                    tf.random_uniform([num_layer_3, num_layer_4], minval=min_sigmoid_3, maxval=max_sigmoid_3),
                    name="W_3")
            with tf.name_scope("Biases_3"):
                self.b3 = tf.Variable(tf.zeros([num_layer_4]), name="b_3")
            with tf.name_scope("Output_3"):
                self.y3 = tf.matmul(self.activations2, self.W3) + self.b3
            with tf.name_scope("BN_3"):
                self.bn3 = tf.contrib.layers.batch_norm(self.y3,
                                                        center=True, scale=True,
                                                        is_training=True,
                                                        scope='bn3')
            with tf.name_scope("Activation_3"):
                self.activations3 = tf.nn.tanh(self.bn3, name='activation3')

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


class td_prediction_simple_V7(object):
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

        with tf.name_scope("Dense_Layer_first"):
            self.x = tf.placeholder(tf.float32, [None, num_layer_1], name="x_1")
            with tf.name_scope("Weight_1"):
                self.W1 = tf.get_variable('w1_xaiver', [num_layer_1, num_layer_2],
                                          initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope("Biases_1"):
                self.b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
            with tf.name_scope("Output_1"):
                self.y1 = tf.matmul(self.x, self.W1) + self.b1
            with tf.name_scope("BN_1"):
                self.bn1 = tf.contrib.layers.batch_norm(self.y1,
                                                        center=True, scale=True,
                                                        is_training=True,
                                                        scope='bn1')
            with tf.name_scope("Activation_1"):
                self.activations1 = tf.nn.relu(self.bn1, name='activation1')

        with tf.name_scope("Dense_Layer_second"):
            with tf.name_scope("Weight_2"):
                self.W2 = tf.get_variable('w2_xaiver', [num_layer_2, num_layer_3],
                                          initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope("Biases_2"):
                self.b2 = tf.Variable(tf.zeros([num_layer_3]), name="b_2")
            with tf.name_scope("Output_2"):
                self.y2 = tf.matmul(self.activations1, self.W2) + self.b2
            with tf.name_scope("BN_2"):
                self.bn2 = tf.contrib.layers.batch_norm(self.y2,
                                                        center=True, scale=True,
                                                        is_training=True,
                                                        scope='bn2')
            with tf.name_scope("Activation_2"):
                self.activations2 = tf.nn.relu(self.bn2, name='activation2')

        with tf.name_scope("Dense_Layer_third"):
            with tf.name_scope("Weight_3"):
                self.W3 = tf.get_variable('w3_xaiver', [num_layer_3, num_layer_4],
                                          initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope("Biases_3"):
                self.b3 = tf.Variable(tf.zeros([num_layer_4]), name="b_3")
            with tf.name_scope("Output_3"):
                self.y3 = tf.matmul(self.activations2, self.W3) + self.b3
            with tf.name_scope("BN_3"):
                self.bn3 = tf.contrib.layers.batch_norm(self.y3,
                                                        center=True, scale=True,
                                                        is_training=True,
                                                        scope='bn3')
            with tf.name_scope("Activation_3"):
                self.activations3 = tf.nn.relu(self.bn3, name='activation3')

        with tf.name_scope("Dense_Layer_fourth"):
            with tf.name_scope("Weight_4"):
                self.W4 = tf.get_variable('w4_xaiver', [num_layer_4, num_layer_5],
                                          initializer=tf.contrib.layers.xavier_initializer())
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


def get_next_test_event():
    """
    retrieve next event from sport data
    :return:
    """
    state = []
    reward = 0
    terminal = 0
    return state, reward, terminal


def get_training_batch(s_t0, state, reward, train_number, train_len):
    """
    combine training data to a batch
    :return: [last_state_of_batch, batch, time_series]
    """
    batch_return = []
    current_batch_length = 0
    while current_batch_length < BATCH_SIZE:
        s_t1 = state[train_number]
        r_t0 = reward[train_number - 1]
        r_t1 = reward[train_number]
        train_number += 1
        if train_number + 1 == train_len:
            batch_return.append((s_t0, r_t0, s_t1, 0))
            batch_return.append((s_t1, r_t1, s_t1, 1))
            break
        batch_return.append((s_t0, r_t0, s_t1, 0))
        current_batch_length += 1
        s_t0 = s_t1

    return s_t0, batch_return, train_number


def write_list_txt(data_list):
    with open(LOG_DIR + '/avg_cost_record.txt', 'wb') as f:
        for data in data_list:
            f.write(data)


def train_network(sess, model, print_parameters=False):
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
    if model_train_continue:
        checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORK)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    cost_all_record = []

    # iterate over the training data
    # for i in range(0, ITERATE_NUM):
    while True:
        if converge_flag:
            break
        elif game_number >= number_of_total_game * ITERATE_NUM:
            break
        else:
            converge_flag = True

        cost_per_iter_record = []
        for dir_game in DIR_GAMES_ALL:
            game_number += 1
            game_files = os.listdir(DATA_STORE + "/" + dir_game)
            for filename in game_files:
                if filename.startswith("reward"):
                    reward_name = filename
                elif filename.startswith("state"):
                    state_name = filename

            reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + reward_name)
            try:
                reward = (reward['reward'][0]).tolist()
            except:
                print "\n" + dir_game
                continue
            reward_count = sum(reward)
            state = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_name)
            state = state['state']
            print ("\n load file" + str(dir_game) + " success")
            print ("reward number" + str(reward_count))
            if len(state) != len(reward):
                raise Exception('state length does not equal to reward length')

            train_len = len(state)
            train_number = 0

            # start training
            s_t0 = state[0]
            r_t0 = reward[0]
            train_number += 1

            if not Random_select:
                while True:
                    try:

                        s_tl, batch, train_number = get_training_batch(s_t0, state,
                                                                       reward,
                                                                       train_number,
                                                                       train_len)

                    except:
                        print "\n game:" + dir_game + " train number:" + str(train_number)
                        raise IndexError("get_training_batch wrong")

                    # get the batch variables
                    s_t_batch = [d[0] for d in batch]
                    r_t_batch = [d[1] for d in batch]
                    s_t1_batch = [d[2] for d in batch]

                    y_batch = []

                    readout_t1_batch = model.read_out.eval(feed_dict={model.x: s_t1_batch})  # get value of s

                    for i in range(0, len(batch)):
                        terminal = batch[i][3]
                        # if terminal, only equals reward
                        if terminal:
                            y_batch.append(float(r_t_batch[i]))
                            break
                        else:
                            y_batch.append(r_t_batch[i] + GAMMA * ((readout_t1_batch[i]).tolist())[0])

                    if MODEL_TYPE == "V5":
                        [global_step, learning_rate, diff_v, cost_out, summary_train, _] = sess.run(
                            [model.global_step, model.learning_rate, model.diff_v, model.cost, merge, model.train_step],
                            feed_dict={model.y: y_batch, model.x: s_t_batch})
                    else:
                        [diff_v, cost_out, summary_train, _] = sess.run(
                            [model.diff_v, model.cost, merge, model.train_step],
                            feed_dict={model.y: y_batch, model.x: s_t_batch})

                    if diff_v > 0.01:
                        converge_flag = False
                    global_counter += 1
                    cost_per_iter_record.append(cost_out)
                    train_writer.add_summary(summary_train, global_step=global_counter)
                    # update the old values
                    s_t0 = s_tl

                    # print info
                    if terminal or ((train_number - 1) / BATCH_SIZE) % 5 == 1:
                        print ("TIMESTEP:", train_number, "Game:", game_number)
                        print(str((min(readout_t1_batch)[0], max(readout_t1_batch)[0])))

                        if MODEL_TYPE == "V5":
                            print ("cost of the network is: " + str(cost_out) + " with learning rate: " + str(
                                learning_rate) + " and global step: " + str(global_step))
                        else:
                            print ("cost of the network is: " + str(cost_out))

                    if terminal:
                        # save progress after a game
                        saver.save(sess, SAVED_NETWORK + '/' + SPORT + '-game-', global_step=game_number)
                        break
            else:
                raise ValueError("Haven't define for random yet")

        cost_per_iter_average = sum(cost_per_iter_record) / float(len(cost_per_iter_record))
        cost_all_record.append(
            "Iter:" + str(game_number / number_of_total_game) + " avg_cost:" + str(cost_per_iter_average))

    # write_list_txt(cost_per_iter_record)
    train_writer.close()


def train_start():
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.isdir(SAVED_NETWORK):
        os.mkdir(SAVED_NETWORK)

    sess = tf.InteractiveSession()
    if MODEL_TYPE == "V1":
        nn = td_prediction_simple()
    elif MODEL_TYPE == "V2":
        nn = td_prediction_simple_V2()
    elif MODEL_TYPE == "V3":
        nn = td_prediction_simple_V3()
    elif MODEL_TYPE == "V4":
        nn = td_prediction_simple_V4()
    elif MODEL_TYPE == "V5":
        nn = td_prediction_simple_V5()
    elif MODEL_TYPE == "V6":
        nn = td_prediction_simple_V6()
    elif MODEL_TYPE == "V7":
        nn = td_prediction_simple_V7()
    else:
        raise ValueError("Unclear model type")
    train_network(sess, nn)


if __name__ == '__main__':
    train_start()