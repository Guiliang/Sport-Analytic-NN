import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf

import td_three_prediction_lstm_v_correct_cut_together

FEATURE_TYPE = 5
MODEL_TYPE = "v3"
ITERATE_NUM = 30
BATCH_SIZE = 32
DRAW_TARGET = "Q_home"
if_correct_velocity = "_v_correct_"
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
    nn = td_three_prediction_lstm_v_correct_cut_together.td_prediction_lstm_V3()
else:
    raise ValueError("Unclear model type")

SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/hybrid_sl_saved_NN/Scale-three-cut_together_saved_networks_feature{0}_batch{1}_iterate{2}_lr{3}_{4}{5}".format(
    str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate), str(MODEL_TYPE), if_correct_velocity)

TACTIC_TYPE = 'forecheck'
SIMULATION_TYPE = 'entire_tactic_simulation'
HIS_ACTION_TYPE = ['lpr', 'carry', 'dumpin', 'lpr', 'check', 'pass', 'block', 'pass', 'shot']
HIS_ACTION_TYPE_COORD = [{'xAdjCoord': -50, 'yAdjCoord': -20, 'home': 1, 'away': 0},
                         {'xAdjCoord': -25, 'yAdjCoord': -20, 'home': 1, 'away': 0},
                         {'xAdjCoord': 0, 'yAdjCoord': -20, 'home': 1, 'away': 0},
                         {'xAdjCoord': -90, 'yAdjCoord': 25, 'home': 0, 'away': 1},
                         {'xAdjCoord': 90, 'yAdjCoord': -22.5, 'home': 1, 'away': 0},
                         {'xAdjCoord': -90, 'yAdjCoord': 20, 'home': 0, 'away': 1},
                         {'xAdjCoord': 80, 'yAdjCoord': -25, 'home': 1, 'away': 0},
                         {'xAdjCoord': 80, 'yAdjCoord': -10, 'home': 1, 'away': 0},
                         {'xAdjCoord': 85, 'yAdjCoord': 0, 'home': 1, 'away': 0},
                         ]


def nn_simulation(tactic_data, trace_length):
    saver = tf.train.Saver()
    sess_nn.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(SIMPLE_SAVED_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess_nn, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)

    else:
        print SIMPLE_SAVED_NETWORK_PATH
        raise Exception("can't restore network")

    readout_values = model_nn.read_out.eval(
        feed_dict={model_nn.rnn_input: tactic_data, model_nn.trace_lengths: trace_length})

    plot_value(readout_values)


def plot_value(readout_values):
    value_home = readout_values[:, 0]
    value_away = readout_values[:, 1]
    value_end = readout_values[:, 2]
    x = range(1, len(value_home)+1)

    plt.plot(x, value_home, label="value_home")
    plt.plot(x, value_away, label="value_away")
    plt.plot(x, value_end, label="value_end")
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    # nn_image_save_dir = "./image/{7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}.png".format(
    #     ACTION_TYPE, str(HIS_ACTION_TYPE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate),
    #     str(MODEL_TYPE), DRAW_TARGET, if_correct_velocity)
    # nn_half_image_save_dir = "./image/right half {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}.png".format(
    #     ACTION_TYPE, str(HIS_ACTION_TYPE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate),
    #     str(MODEL_TYPE), DRAW_TARGET, if_correct_velocity)
    # blend_image_save_dir = "./image/blend {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}.png".format(
    #     ACTION_TYPE, str(HIS_ACTION_TYPE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate),
    #     str(MODEL_TYPE), DRAW_TARGET, if_correct_velocity)
    # blend_half_image_save_dir = "./image/blend right half {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}.png".format(
    #     ACTION_TYPE, str(HIS_ACTION_TYPE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate),
    #     str(MODEL_TYPE), DRAW_TARGET, if_correct_velocity)
    sess_nn = tf.InteractiveSession()
    model_nn = nn
    tactic_data_list = []
    tactic_trace_length_list = []

    for action_index in range(0, len(HIS_ACTION_TYPE)):
        tactic_data_path = "/cs/oschulte/Galen/Hockey-data-entire/Simulation-data-feature5_v_correct_/entire_tactic_simulation/LSTM_" + SIMULATION_TYPE + "-" + TACTIC_TYPE + '-' + str(
            HIS_ACTION_TYPE[0:action_index + 1]) + "-feature" + str(FEATURE_TYPE) + ".mat"
        tactic_trace_length = action_index + 1
        tactic_data = sio.loadmat(tactic_data_path)
        tactic_data = (tactic_data['tactic_data'])
        tactic_data_list.append(tactic_data)
        tactic_trace_length_list.append(tactic_trace_length)

    nn_simulation(np.asarray(tactic_data_list), np.asarray(tactic_trace_length_list))
