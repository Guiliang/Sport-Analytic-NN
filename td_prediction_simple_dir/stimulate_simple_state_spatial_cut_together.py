import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import seaborn as sns

import td_state_prediction_simple_cut_together

FEATURE_TYPE = 5
ACTION_TYPE = "shot"
STIMULATE_TYPE = "position"
MODEL_TYPE = "V3"
ITERATE_NUM = 50
BATCH_SIZE = 32
ISHOME = True
pre_initialize = False
if pre_initialize:
    pre_initialize_situation = "-pre_initialize"
    pre_initialize_save = "_pre_initialize"
else:
    pre_initialize_situation = ""
    pre_initialize_save = ""

learning_rate = 1e-6
if learning_rate == 1e-5:
    learning_rate_write = 5
elif learning_rate == 1e-4:
    learning_rate_write = 4
elif learning_rate == 1e-6:
    learning_rate_write = 6

if MODEL_TYPE == "V3":
    nn = td_state_prediction_simple_cut_together.td_prediction_simple_V3()
else:
    raise ValueError("Unclear model type")

SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/Scale-state-cut_saved_entire_together_networks_feature{0}_batch{1}_iterate{2}_lr{3}-NEG_REWARD_GAMMA1_{4}-Sequenced{5}".format(
    str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM),str(learning_rate), str(MODEL_TYPE), pre_initialize_situation)

if ISHOME:
    SIMULATION_DATA_PATH = "/cs/oschulte/Galen/Hockey-data-entire/Simulation-data-feature{0}/entire_state_spatial_simulation/Home_entire_state_spatial_simulation-feature{0}.mat".format(
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
    ax = sns.heatmap(value_spatial_home, xticklabels=False, yticklabels=False, cmap="RdYlBu_r", vmin=0.50, vmax=0.75)
    plt.xlabel('XAdjcoord', fontsize=16)
    plt.ylabel('YAdjcoord', fontsize=16)
    plt.title("Heatmap of V_home team with TD simple", fontsize=20)
    sns.plt.show()


if __name__ == '__main__':
    sess_nn = tf.InteractiveSession()
    model_nn = nn
    stimulate_value_home = nn_simulation(SIMULATION_DATA_PATH, SIMPLE_SAVED_NETWORK_PATH)
