import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf

from td_prediction_simple_dir import td_prediction_simple

FEATURE_TYPE = 5
ISHOME = True
ACTION_TYPE = "shot"
STIMULATE_TYPE = "position"

SIMPLE_HOME_SAVED_NETWORK_PATH = "/home/gla68/PycharmProjects/Sport-Analytic-NN/td_prediction_simple_dir/saved_NN/saved_Home_networks_feature" + str(
    FEATURE_TYPE) + "_batch16_iterate2Converge-NEG_REWARD_GAMMA1_V3-Sequenced"
SIMPLE_AWAY_SAVED_NETWORK_PATH = "/home/gla68/PycharmProjects/Sport-Analytic-NN/td_prediction_simple_dir/saved_NN/saved_Away_networks_feature" + str(
    FEATURE_TYPE) + "_batch16_iterate2Converge-NEG_REWARD_GAMMA1_V3-Sequenced"

SIMULATION_HOME_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature" + str(
    FEATURE_TYPE) + "/" + STIMULATE_TYPE + "_simulation/" + STIMULATE_TYPE + "_simulation-" + ACTION_TYPE + "-feature" + str(
    FEATURE_TYPE) + "-[][].mat"
SIMULATION_AWAY_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature" + str(
    FEATURE_TYPE) + "/" + STIMULATE_TYPE + "_simulation/Away_" + STIMULATE_TYPE + "_simulation-" + ACTION_TYPE + "-feature" + str(
    FEATURE_TYPE) + "-[][].mat"

RNN_SAVED_NETWORK_PATH = "./saved_NN/"


# SIMPLE_SAVED_NETWORK_PATH = "./saved_networks_feature2_FORWARD"


# def rnn_simulation():
#     sess_nn = tf.InteractiveSession()
#     rnn_input_nn, read_out_nn, y_nn, train_step_nn, cost_nn = td_prediction_RNN.create_network_RNN_type2()
#
#     simulate_data = sio.loadmat(SIMULATION_DATA_PATH)
#     simulate_data = simulate_data['simulate_data']
#
#     saver = tf.train.Saver()
#     sess_nn.run(tf.global_variables_initializer())
#
#     checkpoint = tf.train.get_checkpoint_state(RNN_SAVED_NETWORK_PATH)
#     if checkpoint and checkpoint.model_checkpoint_path:
#         saver.restore(sess_nn, checkpoint.model_checkpoint_path)
#         print("Successfully loaded:", checkpoint.model_checkpoint_path)
#     else:
#         print("Could not find old network weights")
#
#     return None


def nn_simulation(SIMULATION_DATA_PATH, SIMPLE_SAVED_NETWORK_PATH):
    sess_nn = tf.InteractiveSession()
    model_nn = td_prediction_simple.td_prediction_simple_V3()

    simulate_data = sio.loadmat(SIMULATION_DATA_PATH)
    simulate_data = (simulate_data['simulate_data'])

    saver = tf.train.Saver()
    sess_nn.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(SIMPLE_SAVED_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess_nn, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        raise Exception("can't restore network")

    readout_t1_batch = model_nn.read_out.eval(feed_dict={model_nn.x: simulate_data})
    # print (readout_t1_batch.tolist())
    #
    draw_value_over_position(readout_t1_batch)
    #
    # print(max(readout_t1_batch))
    # print(min(readout_t1_batch))

    return readout_t1_batch.tolist()


def draw_value_over_position(y):
    max_data = y.max()
    min_data = y.min()
    # max_data = (max(y))[0]
    # min_data = (min(y))[0]
    scale = float(80) / (max_data - min_data)
    y_list = y.tolist()
    y_deal = []
    y_deal
    for y_data in y_list:
        # y_data_deal = ((y_data[0] - min_data) * scale) - 40
        y_deal.append(y_data[0])

    if STIMULATE_TYPE == "angel":
        x = np.arange(-0, 360, float(360) / 120)
    elif STIMULATE_TYPE == "position":
        x = np.arange(-100, 100, 2)
    plt.plot(x, y_deal)
    # img = imread('./hockey-field.png')
    # plt.imshow(img, extent=[-100, 100, -50, 50])
    plt.show()
    return None


if __name__ == '__main__':
    sess_nn = tf.InteractiveSession()
    model_nn = td_prediction_simple.td_prediction_simple_V3()
    stimulate_value_home = nn_simulation(SIMULATION_HOME_DATA_PATH, SIMPLE_HOME_SAVED_NETWORK_PATH)
    stimulate_value_away = nn_simulation(SIMULATION_AWAY_DATA_PATH, SIMPLE_AWAY_SAVED_NETWORK_PATH)
    stimulate_value_away_abs = map(abs, stimulate_value_away)
    stimulate_value_rate = [float(c) / float(d) for c, d in zip(stimulate_value_home, stimulate_value_away_abs)]
    draw_value_over_position(stimulate_value_rate)
