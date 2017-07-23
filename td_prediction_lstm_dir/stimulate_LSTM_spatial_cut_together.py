import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import seaborn as sns

import td_prediction_lstm_cut_together

FEATURE_TYPE = 5
ACTION_TYPE = "shot-pass"
STIMULATE_TYPE = "position"
MODEL_TYPE = "v3"
ITERATE_NUM = 30
BATCH_SIZE = 8
ISHOME = True
HIS_ACTION_TYPE = ['reception', 'pass', 'reception']
pre_initialize = False
if pre_initialize:
    pre_initialize_situation = "-pre_initialize"
    pre_initialize_save = "_pre_initialize"
else:
    pre_initialize_situation = ""
    pre_initialize_save = ""

learning_rate = 1e-5
if learning_rate == 1e-5:
    learning_rate_write = 5
elif learning_rate == 1e-4:
    learning_rate_write = 4

if MODEL_TYPE == "v3":
    nn = td_prediction_lstm_cut_together.td_prediction_lstm_V3()
else:
    raise ValueError("Unclear model type")

SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/hybrid_sl_saved_NN/Scale-cut_together_saved_networks_feature{0}_batch{1}_iterate{2}_lr{3}_{4}".format(
    str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate), str(MODEL_TYPE))

if ISHOME:
    SIMULATION_DATA_PATH = "/cs/oschulte/Galen/Hockey-data-entire/Simulation-data-feature{0}/entire_spatial_simulation/LSTM_Home_entire_spatial_simulation-{1}-{2}-feature{0}.mat".format(
        str(FEATURE_TYPE), str(ACTION_TYPE), str(HIS_ACTION_TYPE))


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
        trace_length = np.ones(len(x_coord_states))*len(ACTION_TYPE.split('-'))
        readout_x_coord_values = model_nn.read_out.eval(feed_dict={model_nn.rnn_input: x_coord_states, model_nn.trace_lengths: trace_length})
        value_spatial_home.append((readout_x_coord_values[:, 0]).tolist())
        value_spatial_away.append((readout_x_coord_values[:, 1]).tolist())

    print "heat map"
    sns.set()
    ax = sns.heatmap(value_spatial_home, xticklabels=False, yticklabels=False, cmap="RdYlBu_r")
    plt.xlabel('XAdjcoord', fontsize=16)
    plt.ylabel('YAdjcoord', fontsize=16)
    plt.title("Q_home {0} with Dynamic LSTM".format(ACTION_TYPE), fontsize=20)
    sns.plt.show()


if __name__ == '__main__':
    sess_nn = tf.InteractiveSession()
    model_nn = nn
    stimulate_value_home = nn_simulation(SIMULATION_DATA_PATH, SIMPLE_SAVED_NETWORK_PATH)
