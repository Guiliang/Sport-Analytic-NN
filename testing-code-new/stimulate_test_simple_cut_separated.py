import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf

import td_prediction_simple_testing

FEATURE_TYPE = 5
ACTION_TYPE = "shot"
STIMULATE_TYPE = "position"
MODEL_TYPE = "V3"
ITERATE_NUM = 25
BATCH_SIZE = 8
DATA_SIZE = 100

if MODEL_TYPE == "V1":
    nn = td_prediction_simple_testing.td_prediction_simple()
elif MODEL_TYPE == "V2":
    nn = td_prediction_simple_testing.td_prediction_simple_V2()
elif MODEL_TYPE == "V3":
    nn = td_prediction_simple_testing.td_prediction_simple_V3()
elif MODEL_TYPE == "V4":
    nn = td_prediction_simple_testing.td_prediction_simple_V4()
elif MODEL_TYPE == "V5":
    nn = td_prediction_simple_testing.td_prediction_simple_V5()
elif MODEL_TYPE == "V6":
    nn = td_prediction_simple_testing.td_prediction_simple_V6()
elif MODEL_TYPE == "V7":
    nn = td_prediction_simple_testing.td_prediction_simple_V7()
else:
    raise ValueError("Unclear model type")

SIMPLE_HOME_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/Test" + str(
    DATA_SIZE) + "-cut_saved_entire_Home_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "-NEG_REWARD_GAMMA1_" + MODEL_TYPE + "-Sequenced"
SIMPLE_AWAY_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/Test" + str(
    DATA_SIZE) + "-cut_saved_entire_Away_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "-NEG_REWARD_GAMMA1_" + MODEL_TYPE + "-Sequenced"

SIMULATION_HOME_DATA_PATH = "/cs/oschulte/Galen/Hockey-data-entire/Simulation-data-feature" + str(
    FEATURE_TYPE) + "/" + STIMULATE_TYPE + "_simulation/Home_" + STIMULATE_TYPE + "_simulation-" + ACTION_TYPE + "-feature" + str(
    FEATURE_TYPE) + "-[][].mat"
SIMULATION_AWAY_DATA_PATH = "/cs/oschulte/Galen/Hockey-data-entire/Simulation-data-feature" + str(
    FEATURE_TYPE) + "/" + STIMULATE_TYPE + "_simulation/Away_" + STIMULATE_TYPE + "_simulation-" + ACTION_TYPE + "-feature" + str(
    FEATURE_TYPE) + "-[][].mat"


def nn_simulation(SIMULATION_DATA_PATH, SIMPLE_SAVED_NETWORK_PATH):
    simulate_data = sio.loadmat(SIMULATION_DATA_PATH)
    simulate_data = (simulate_data['simulate_data'])

    saver = tf.train.Saver()
    sess_nn.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(SIMPLE_SAVED_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess_nn, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print SIMPLE_SAVED_NETWORK_PATH
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
    try:
        y_list = y.tolist()
        y_deal = []
        for y_data in y_list:
            # y_data_deal = ((y_data[0] - min_data) * scale) - 40
            y_deal.append(y_data[0])
    except:
        y_deal = y

    if STIMULATE_TYPE == "angel":
        x = np.arange(-0, 360, float(360) / 120)
    elif STIMULATE_TYPE == "position":
        x = np.arange(-100, 100, 2)
    plt.figure()
    plt.plot(x, y_deal)
    # img = imread('./hockey-field.png')
    # plt.imshow(img, extent=[-100, 100, -50, 50])
    plt.show()
    return None


if __name__ == '__main__':
    sess_nn = tf.InteractiveSession()
    model_nn = nn
    stimulate_value_home = nn_simulation(SIMULATION_HOME_DATA_PATH, SIMPLE_HOME_SAVED_NETWORK_PATH)
    stimulate_value_away = nn_simulation(SIMULATION_AWAY_DATA_PATH, SIMPLE_AWAY_SAVED_NETWORK_PATH)
    stimulate_value_home = [value[0] for value in stimulate_value_home]
    stimulate_value_away = [value[0] for value in stimulate_value_away]
    stimulate_value_away_abs = map(abs, stimulate_value_away)
    stimulate_value_away_abs.reverse()
    stimulate_value_rate = [float(c) / float(d) for c, d in zip(stimulate_value_home, stimulate_value_away_abs)]
    draw_value_over_position(stimulate_value_rate)
