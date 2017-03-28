import scipy.io as sio
import tensorflow as tf
import os
import numpy as np
import math

"""
train the home team and away team together, use a feature to represent it.
"""
feature_num = 12
GAMMA = 0.99  # decay rate of past observations
BATCH_SIZE = 32  # size of minibatch
SPORT = "NHL"
DATA_STORE = "/home/gla68/Documents/Hockey-data/Hockey-Training-All-feature2-scale"
DIR_GAMES_ALL = os.listdir(DATA_STORE)
LOG_DIR = "./log_train_feature2_FORWARD"
SAVED_NETWORK = "./saved_networks_feature2_FORWARD"
FORWARD_REWARD_MODE = False


def create_network():
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
        x = tf.placeholder(tf.float32, [None, num_layer_1], name="x_1")
        with tf.name_scope("Weight_1"):
            W1 = tf.Variable(tf.random_uniform([num_layer_1, num_layer_2], minval=min_sigmoid_1, maxval=max_sigmoid_1),
                             name="W_1")
        with tf.name_scope("Biases_1"):
            b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
        with tf.name_scope("Output_1"):
            y1 = tf.matmul(x, W1) + b1
        with tf.name_scope("Activation_1"):
            activations = tf.nn.sigmoid(y1, name='activation')
            tf.summary.histogram('activation_1', activations)

    # to debug the network
    W1_print = tf.Print(W1, [W1], message="W1 is:", summarize=40)
    y1_print = tf.Print(y1, [y1], message="y1 is:", summarize=40)
    b1_print = tf.Print(b1, [b1], message="b1 is:", summarize=40)

    with tf.name_scope("Dense_Layer_second"):
        with tf.name_scope("Weight_2"):
            W2 = tf.Variable(tf.random_uniform([num_layer_2, num_layer_3], minval=min_sigmoid_2, maxval=max_sigmoid_2),
                             name="W_2")
        with tf.name_scope("Biases_1"):
            b2 = tf.Variable(tf.zeros([num_layer_3]), name="b_2")
        with tf.name_scope("Output_2"):
            read_out = tf.matmul(activations, W2) + b2
            tf.summary.histogram('output_2', activations)

    # to debug the network
    W2_print = tf.Print(W2, [W2], message="W2 is:", summarize=40)
    y2_print = tf.Print(read_out, [read_out], message="y2 is:", summarize=40)
    b2_print = tf.Print(b2, [b2], message="b2 is:", summarize=40)

    # define the cost function
    y = tf.placeholder("float", [None])

    with tf.name_scope("cost"):
        readout_action = tf.reduce_sum(read_out,
                                       reduction_indices=1)  # Computes the sum of elements across dimensions of a tensor.
        cost = tf.reduce_mean(tf.square(y - readout_action))  # square means
    tf.summary.histogram('cost', cost)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    # train_step = tf.train.AdadeltaOptimizer().minimize(cost)

    return x, read_out, y, train_step, cost, W1_print, y1_print, b1_print, W2_print, y2_print, b2_print


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
        r_t1 = reward[train_number]
        r_t0 = reward[train_number - 1]
        train_number += 1
        if train_number + 1 == train_len:
            if FORWARD_REWARD_MODE:
                batch_return.append((s_t0, r_t1, s_t1, 1))
            else:
                batch_return.append((s_t0, r_t0, s_t1, 1))
            break
        if FORWARD_REWARD_MODE:
            batch_return.append((s_t0, r_t1, s_t1, 0))
        else:
            batch_return.append((s_t0, r_t0, s_t1, 0))
        current_batch_length += 1
        s_t0 = s_t1

    return s_t0, batch_return, train_number


def build_training_batch(state, reward):
    """
    build batches
    :param state:
    :param reward:
    :return:
    """
    batch_return = []
    batch_number = len(state)
    s_t0 = state[0]
    for num in range(1, batch_number):
        s_t1 = state[num]
        r_t1 = reward[num]
        if num == batch_number - 1:
            terminal = 1
        else:
            terminal = 0
        batch_return.append({'state_0': s_t0, 'reward': r_t1, 'state_1': s_t1, 'terminal': terminal})
        s_t0 = s_t1
    return batch_return


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
    merge = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(LOG_DIR, sess.graph)
    sess.run(tf.global_variables_initializer())
    # checkpoint = tf.train.get_checkpoint_state("./saved_networks/")
    # if checkpoint and checkpoint.model_checkpoint_path:
    #     saver.restore(sess, checkpoint.model_checkpoint_path)
    #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
    # else:
    #     print("Could not find old network weights")

    # iterate over the training data
    for i in range(0, 3):
        for dir_game in DIR_GAMES_ALL:
            game_number += 1
            game_files = os.listdir(DATA_STORE + "/" + dir_game)
            for filename in game_files:
                if filename.startswith("reward"):
                    reward_name = filename
                elif filename.startswith("state"):
                    state_name = filename

            reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + reward_name)
            reward = (reward['reward'][0]).tolist()
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
                        y_batch.append(float(r_t_batch[i]))
                        break
                    else:
                        y_batch.append(r_t_batch[i] + GAMMA * ((readout_t1_batch[i]).tolist())[0])

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
                    saver.save(sess, SAVED_NETWORK + '/' + SPORT + '-game-', global_step=game_number)
                    break

    train_writer.close()


# def compute_state_q(x, readout):
#     """
#     print testing data
#     :param x: network input placeholder
#     :param readout:
#     :return:
#     """
#     t = 0
#     terminal = 0
#     if terminal != 1:
#         s_t, r_t, a_t, terminal = get_next_test_event()
#         readout_t = readout.eval(feed_dict={x: s_t})
#         q_t = readout_t[a_t]
#         print ("time", t, "\tstate:", s_t, "\t q_value:", q_t)


def train_start():
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.isdir(SAVED_NETWORK):
        os.mkdir(SAVED_NETWORK)

    sess = tf.InteractiveSession()
    x, read_out, y, train_step, cost, W1_print, y1_print, b1_print, W2_print, y2_print, b2_print = create_network()
    train_network(sess, x, read_out, y, train_step, cost, W1_print, y1_print, b1_print, W2_print, y2_print, b2_print)


if __name__ == '__main__':
    train_start()
