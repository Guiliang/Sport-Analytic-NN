import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import seaborn as sns

import td_state_prediction_simple_cut_together
import td_prediction_simple_cut_together

FEATURE_TYPE = 5
ACTION_TYPE = "block"
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
    state_graph = tf.Graph()
    with state_graph.as_default():
        state_nn = td_state_prediction_simple_cut_together.td_prediction_simple_V3()

    q_graph = tf.Graph()
    with q_graph.as_default():
        q_nn = td_prediction_simple_cut_together.td_prediction_simple_V3()
else:
    raise ValueError("Unclear model type")

STATE_SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/Scale-state-cut_saved_entire_together_networks_feature{0}_batch{1}_iterate{2}_lr{3}-NEG_REWARD_GAMMA1_{4}-Sequenced{5}".format(
    str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate), str(MODEL_TYPE), pre_initialize_situation)
Q_SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/Scale-cut_saved_entire_together_networks_feature{0}_batch{1}_iterate{2}-NEG_REWARD_GAMMA1_{3}-Sequenced{4}".format(
    str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(MODEL_TYPE), pre_initialize_situation)

if ISHOME:
    STATE_SIMULATION_DATA_PATH = "/cs/oschulte/Galen/Hockey-data-entire/Simulation-data-feature{0}/entire_state_spatial_simulation/Home_entire_state_spatial_simulation-feature{0}.mat".format(
        str(FEATURE_TYPE))
    Q_SIMULATION_DATA_PATH = "/cs/oschulte/Galen/Hockey-data-entire/Simulation-data-feature{0}/entire_spatial_simulation/Home_entire_spatial_simulation-{1}-feature{0}.mat".format(
        str(FEATURE_TYPE), str(ACTION_TYPE))


def nn_simulation(sess_nn, model_nn, SIMULATION_DATA_PATH, SIMPLE_SAVED_NETWORK_PATH):
    simulate_data = sio.loadmat(SIMULATION_DATA_PATH)
    simulate_data = (simulate_data['simulate_data'])

    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables())

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

    return value_spatial_home


def draw_graph(value):
    print "heat map"
    sns.set()
    ax = sns.heatmap(value, xticklabels=False, yticklabels=False, cmap="RdYlBu_r")
    plt.xlabel('XAdjcoord', fontsize=16)
    plt.ylabel('YAdjcoord', fontsize=16)
    plt.title("Heatmap of home impact with TD simple", fontsize=20)
    sns.plt.show()


def list_minus(list1, list2):
    minus_result = []
    for i in range(0, len(list1)):
        list1_line = list1[i]
        list2_line = list2[i]
        minus_line = []
        for j in range(0, len(list1_line)):
            minus_line.append(list1_line[j] - list2_line[j]/2)
        minus_result.append(minus_line)
    return minus_result


if __name__ == '__main__':
    sess_state = tf.InteractiveSession(graph=state_graph)
    sess_q = tf.InteractiveSession(graph=q_graph)
    with sess_state.as_default():
        with state_graph.as_default():
            state_value_home = nn_simulation(sess_state, state_nn, STATE_SIMULATION_DATA_PATH,
                                             STATE_SIMPLE_SAVED_NETWORK_PATH)
    with sess_q.as_default():
        with q_graph.as_default():
            q_value_home = nn_simulation(sess_q, q_nn, Q_SIMULATION_DATA_PATH, Q_SIMPLE_SAVED_NETWORK_PATH)
    # q_value_home = nn_simulation(sess_nn, q_nn, Q_SIMULATION_DATA_PATH, Q_SIMPLE_SAVED_NETWORK_PATH)
    impact = list_minus(q_value_home, state_value_home)
    draw_graph(impact)
