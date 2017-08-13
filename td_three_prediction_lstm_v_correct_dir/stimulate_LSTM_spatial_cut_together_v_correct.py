import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import seaborn as sns
import cv2

import td_three_prediction_lstm_v_correct_cut_together

FEATURE_TYPE = 5
ACTION_TYPE = "shot"
STIMULATE_TYPE = "position"
MODEL_TYPE = "v3"
ITERATE_NUM = 30
BATCH_SIZE = 32
ISHOME = True
HIS_ACTION_TYPE = []
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
# "Scale-three-cut_together_saved_networks_feature5_batch32_iterate30_lr1e-05_v3_v_correct_"
if ISHOME:
    SIMULATION_DATA_PATH = "/cs/oschulte/Galen/Hockey-data-entire/Simulation-data-feature{0}_v_correct_/entire_spatial_simulation/LSTM_Home_entire_spatial_simulation-{1}-{2}-feature{0}.mat".format(
        str(FEATURE_TYPE), str(ACTION_TYPE), str(HIS_ACTION_TYPE))


def nn_simulation(SIMULATION_DATA_PATH, SIMPLE_SAVED_NETWORK_PATH, nn_image_save_dir, nn_half_image_save_dir):
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

    if DRAW_TARGET == "Q_home":
        value_spatial = value_spatial_home
        if ACTION_TYPE == "shot":
            vmin_set = 0.66
            vmax_set = 0.80
        else:
            vmin_set = 0.510
            vmax_set = 0.680
    elif DRAW_TARGET == "Q_away":
        value_spatial = value_spatial_away
        if ACTION_TYPE == "shot":
            vmin_set = 0.16
            vmax_set = 0.50
        else:
            vmin_set = None
            vmax_set = None
    else:
        raise ValueError("wrong type of DRAW_TARGET")

    print "heat map"
    plt.figure(figsize=(15, 6))
    sns.set()
    ax = sns.heatmap(value_spatial, xticklabels=False, yticklabels=False, cmap="RdYlBu_r", vmin=vmin_set, vmax=vmax_set)
    plt.xlabel('XAdjcoord', fontsize=16)
    plt.ylabel('YAdjcoord', fontsize=16)
    try:
        plt.title("{2} action:{0}-history:{1} with DT-LSTM".format(ACTION_TYPE, str(HIS_ACTION_TYPE), DRAW_TARGET),
                  fontsize=20)
    except:
        plt.title("{2} action:{0}-history:{1} with DT-LSTM".format(ACTION_TYPE, "[]", DRAW_TARGET), fontsize=20)

    # sns.plt.show()
    print nn_image_save_dir
    plt.savefig(nn_image_save_dir)

    value_spatial_home_half = [v[200:402] for v in value_spatial]
    plt.figure(figsize=(15, 12))
    sns.set()
    ax = sns.heatmap(value_spatial_home_half, xticklabels=False, yticklabels=False, cmap="RdYlBu_r", vmin=vmin_set,
                     vmax=vmax_set)
    plt.xlabel('XAdjcoord', fontsize=26)
    plt.ylabel('YAdjcoord', fontsize=26)
    try:
        plt.title("{2} action:{0}, history:{1} \n with DT-LSTM on right rink".format(ACTION_TYPE, str(HIS_ACTION_TYPE),
                                                                                    DRAW_TARGET),
                  fontsize=30)
    except:
        plt.title("{2} action:{0}, history:{1} \n with DT-LSTM on right rink".format(ACTION_TYPE, "[]", DRAW_TARGET),
                  fontsize=30)

    plt.savefig(nn_half_image_save_dir)


def image_blending(value_Img_dir, save_dir, value_Img_half_dir, half_save_dir):
    value_Img = cv2.imread(
        value_Img_dir)
    value_Img_half = cv2.imread(
        value_Img_half_dir)
    background = cv2.imread("./image/hockey-field.png")
    # v_rows, v_cols, v_channels = value_Img.shape
    # v_h_rows, v_h_cols, v_h_channels = value_Img_half.shape

    focus_Img = value_Img[60:540, 188:1118]
    f_rows, f_cols, f_channels = focus_Img.shape
    focus_background = cv2.resize(background, (f_cols, f_rows), interpolation=cv2.INTER_CUBIC)
    blend_focus = cv2.addWeighted(focus_Img, 1, focus_background, 0.5, -255 / 2)
    blend_all = value_Img
    blend_all[60:540, 188:1118] = blend_focus
    # final_rows = v_rows * float(b_rows) / float(f_rows)
    # final_cols = v_cols * float(b_cols) / float(f_cols)
    # blend_all_final = cv2.resize(blend_all, (int(final_cols), int(final_rows)), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('res', focus_Img)
    # cv2.waitKey(0)
    cv2.imwrite(save_dir, blend_all)

    focus_Img_half = value_Img_half[120:1090, 190:1125]
    f_h_rows, f_h_cols, f_h_channels = focus_Img_half.shape
    focus_background_half = cv2.resize(background[:, 899:1798, :], (f_h_cols, f_h_rows), interpolation=cv2.INTER_CUBIC)
    blend_half_focus = cv2.addWeighted(focus_Img_half, 1, focus_background_half, 0.5, -255 / 2)
    blend_half_all = value_Img_half
    blend_half_all[120:1090, 190:1125] = blend_half_focus
    cv2.imwrite(half_save_dir, blend_half_all)


if __name__ == '__main__':
    nn_image_save_dir = "./image/{7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}.png".format(
        ACTION_TYPE, str(HIS_ACTION_TYPE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate),
        str(MODEL_TYPE), DRAW_TARGET, if_correct_velocity)
    nn_half_image_save_dir = "./image/right half {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}.png".format(
        ACTION_TYPE, str(HIS_ACTION_TYPE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate),
        str(MODEL_TYPE), DRAW_TARGET, if_correct_velocity)
    blend_image_save_dir = "./image/blend {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}.png".format(
        ACTION_TYPE, str(HIS_ACTION_TYPE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate),
        str(MODEL_TYPE), DRAW_TARGET, if_correct_velocity)
    blend_half_image_save_dir = "./image/blend right half {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}.png".format(
        ACTION_TYPE, str(HIS_ACTION_TYPE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate),
        str(MODEL_TYPE), DRAW_TARGET, if_correct_velocity)
    sess_nn = tf.InteractiveSession()
    model_nn = nn
    nn_simulation(SIMULATION_DATA_PATH, SIMPLE_SAVED_NETWORK_PATH, nn_image_save_dir, nn_half_image_save_dir)
    image_blending(nn_image_save_dir, blend_image_save_dir, nn_half_image_save_dir, blend_half_image_save_dir)