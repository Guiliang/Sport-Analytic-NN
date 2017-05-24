import scipy.io as sio
import tensorflow as tf
import os

import td_prediction_simple_separated

FEATURE_TYPE = 5
MOTION_TYPE = "lpr"
calibration = True
ISHOME = True

SIMPLE_HOME_SAVED_NETWORK_PATH = "/home/gla68/PycharmProjects/Sport-Analytic-NN/td_prediction_simple_dir/saved_NN/saved_Home_networks_feature" + str(
    FEATURE_TYPE) + "_batch16_iterate2Converge-NEG_REWARD_GAMMA1_V3-Sequenced_bak"
SIMPLE_AWAY_SAVED_NETWORK_PATH = "/home/gla68/PycharmProjects/Sport-Analytic-NN/td_prediction_simple_dir/saved_NN/saved_Away_networks_feature" + str(
    FEATURE_TYPE) + "_batch16_iterate2Converge-NEG_REWARD_GAMMA1_V3-Sequenced_bak"

calibration_store_dir = "/media/gla68/Windows/Hockey-data/calibrate_all_feature_" + str(FEATURE_TYPE)

sess_nn = tf.InteractiveSession()
model_nn = td_prediction_simple_separated.td_prediction_simple_V3()

for calibration_dir_game in os.listdir(calibration_store_dir):
    for file_name in os.listdir(calibration_store_dir + "/" + calibration_dir_game):
        if "training_data_dict_all_value" in file_name:
            calibrate_value_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
        elif "training_data_dict_all_name" in file_name:
            calibrate_name_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
        else:
            raise ValueError("Can't find training value")

        calibrate_value = sio.loadmat(calibrate_value_name)
        calibrate_name = sio.loadmat(calibrate_name_name)

        readout_t1_batch = model_nn.read_out.eval(feed_dict={model_nn.x: calibrate_value})  # get value of s

def check_calibrate_home_away(calibrate_name):


if __name__ == '__main__':
    print "abc"
