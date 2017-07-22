import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import seaborn as sns
import Image
import cv2

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

if MODEL_TYPE == "V3":
    nn = td_prediction_simple_cut_together.td_prediction_simple_V3()
else:
    raise ValueError("Unclear model type")

SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/Scale-cut_saved_entire_together_networks_feature{0}_batch{1}_iterate{2}-NEG_REWARD_GAMMA1_{3}-Sequenced{4}".format(
    str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(MODEL_TYPE), pre_initialize_situation)

if ISHOME:
    SIMULATION_DATA_PATH = "/cs/oschulte/Galen/Hockey-data-entire/Simulation-data-feature{0}/entire_spatial_simulation/Home_entire_spatial_simulation-{1}-feature{0}.mat".format(
        str(FEATURE_TYPE), str(ACTION_TYPE))


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
    plt.figure(figsize=(15, 6))
    sns.set()
    ax = sns.heatmap(value_spatial_home, xticklabels=False, yticklabels=False, cmap="RdYlBu_r")

    plt.xlabel('XAdjcoord', fontsize=16)
    plt.ylabel('YAdjcoord', fontsize=16)
    plt.title("Q_home {0} with TD simple".format(ACTION_TYPE), fontsize=20)
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
    blend_focus = cv2.addWeighted(focus_Img, 1, focus_background, 0.5, -255/2)
    blend_all = value_Img
    blend_all[60:540, 188:1118] = blend_focus
    # final_rows = v_rows * float(b_rows) / float(f_rows)
    # final_cols = v_cols * float(b_cols) / float(f_cols)
    # blend_all_final = cv2.resize(blend_all, (int(final_cols), int(final_rows)), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow('res', blend_all)
    # cv2.waitKey(0)
    cv2.imwrite(save_dir, blend_all)


def try_seaborn():
    uniform_data = np.random.rand(5, 12)
    ax = sns.heatmap(uniform_data)
    sns.plt.show()


if __name__ == '__main__':
    nn_image_save_dir = "./image/Q_home {0} with TD simple.png".format(ACTION_TYPE)
    blend_image_save_dir = "./image/blend Q_home {0} with TD simple.png".format(ACTION_TYPE)
    sess_nn = tf.InteractiveSession()
    model_nn = nn
    stimulate_value_home = nn_simulation(SIMULATION_DATA_PATH, SIMPLE_SAVED_NETWORK_PATH, nn_image_save_dir)
    image_blending(nn_image_save_dir, blend_image_save_dir)
