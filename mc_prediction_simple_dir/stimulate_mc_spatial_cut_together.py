import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import seaborn as sns

import mc_together_prediction_simple_cut

FEATURE_TYPE = 5
ACTION_TYPE = "shot"
STIMULATE_TYPE = "position"
MODEL_TYPE = "V3"
ITERATE_NUM = 50
BATCH_SIZE = 32
ISHOME = True
pre_initialize = True
if pre_initialize:
    pre_initialize_situation = "-pre_initialize"
    pre_initialize_save = "_pre_initialize"
else:
    pre_initialize_situation = ""
    pre_initialize_save = ""

if MODEL_TYPE == "V3":
    nn = mc_together_prediction_simple_cut.td_prediction_simple_V3()
else:
    raise ValueError("Unclear model type")

SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/mc-Scale-cut_saved_entire_together_networks_feature{0}_batch{1}_iterate{2}-NEG_REWARD_GAMMA1_{3}-Sequenced{4}".format(
    str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(MODEL_TYPE), pre_initialize_situation)

if ISHOME:
    SIMULATION_DATA_PATH = "/cs/oschulte/Galen/Hockey-data-entire/Simulation-data-feature{0}/entire_spatial_simulation/Home_entire_spatial_simulation-shot-feature{0}.mat".format(
        str(FEATURE_TYPE))


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

    value_spatial_home = []
    value_spatial_away = []

    for x_coord_states in simulate_data:
        readout_x_coord_values = model_nn.read_out.eval(feed_dict={model_nn.x: x_coord_states})
        value_spatial_home.append((readout_x_coord_values[:, 0]).tolist())
        value_spatial_away.append((readout_x_coord_values[:, 1]).tolist())

    # extent = [-100, 100, -50, 50]
    # plt.clf()
    # plt.imshow(value_spatial_home, extent=extent, cmap='hot', interpolation='nearest')
    # plt.xlabel("XAdjcoord")
    # plt.ylabel("YAdjcoord")
    # plt.title("Heatmap of home team with TD simple")
    # plt.show()

    print "heat map"
    sns.set()
    ax = sns.heatmap(value_spatial_home, xticklabels=False, yticklabels=False, cmap="RdYlBu_r")
    plt.xlabel('XAdjcoord', fontsize=16)
    plt.ylabel('YAdjcoord', fontsize=16)
    plt.title("Heatmap of home team with MC", fontsize=20)
    sns.plt.show()


if __name__ == '__main__':
    sess_nn = tf.InteractiveSession()
    model_nn = nn
    stimulate_value_home = nn_simulation(SIMULATION_DATA_PATH, SIMPLE_SAVED_NETWORK_PATH)
