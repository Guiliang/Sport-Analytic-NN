import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import seaborn as sns
import cv2

import td_prediction_lstm_cut_together

FEATURE_TYPE = 5
ACTION_TYPE = "shot"
STIMULATE_TYPE = "position"
MODEL_TYPE = "v3"
ITERATE_NUM = 30
BATCH_SIZE = 8
ISHOME = True
HIS_ACTION_TYPE = []
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


def nn_simulation(SIMULATION_DATA_PATH, SIMPLE_SAVED_NETWORK_PATH, nn_image_save_dir):
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
        trace_length = np.ones(len(x_coord_states)) * (len(HIS_ACTION_TYPE) + 1)
        readout_x_coord_values = model_nn.read_out.eval(
            feed_dict={model_nn.rnn_input: x_coord_states, model_nn.trace_lengths: trace_length})
        value_spatial_home.append((readout_x_coord_values[:, 0]).tolist())
        value_spatial_away.append((readout_x_coord_values[:, 1]).tolist())

    print "heat map"
    plt.figure(figsize=(15, 6))
    sns.set()
    ax = sns.heatmap(value_spatial_home, xticklabels=False, yticklabels=False, cmap="RdYlBu_r")
    plt.xlabel('XAdjcoord', fontsize=16)
    plt.ylabel('YAdjcoord', fontsize=16)
    plt.title("Q_home {0}-{1} with Dynamic LSTM".format(ACTION_TYPE, str(HIS_ACTION_TYPE)), fontsize=20)
    # sns.plt.show()
    plt.savefig(nn_image_save_dir)


def image_blending(value_Img_dir, save_dir):
    value_Img = cv2.imread(
        value_Img_dir)
    background = cv2.imread("./image/hockey-field.png")
    v_rows, v_cols, v_channels = value_Img.shape
    b_rows, b_cols, b_channels = background.shape
    print (v_rows, v_cols)

    focus_Img = value_Img[60:540, 188:1118]
    f_rows, f_cols, f_channels = focus_Img.shape
    focus_background = cv2.resize(background, (f_cols, f_rows), interpolation=cv2.INTER_CUBIC)
    blend_focus = cv2.addWeighted(focus_Img, 1, focus_background, 0.5, -255 / 2)
    blend_all = value_Img
    blend_all[60:540, 188:1118] = blend_focus
    # final_rows = v_rows * float(b_rows) / float(f_rows)
    # final_cols = v_cols * float(b_cols) / float(f_cols)
    # blend_all_final = cv2.resize(blend_all, (int(final_cols), int(final_rows)), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow('res', blend_all)
    # cv2.waitKey(0)
    cv2.imwrite(save_dir, blend_all)


if __name__ == '__main__':
    nn_image_save_dir = "./image/Q_home {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}.png".format(ACTION_TYPE, str(HIS_ACTION_TYPE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate), str(MODEL_TYPE))
    blend_image_save_dir = "./image/blend Q_home {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}.png".format(ACTION_TYPE, str(HIS_ACTION_TYPE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate), str(MODEL_TYPE))
    sess_nn = tf.InteractiveSession()
    model_nn = nn
    stimulate_value_home = nn_simulation(SIMULATION_DATA_PATH, SIMPLE_SAVED_NETWORK_PATH, nn_image_save_dir)
    image_blending(nn_image_save_dir, blend_image_save_dir)
