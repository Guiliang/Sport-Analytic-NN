import scipy.io as sio
import tensorflow as tf
import os
import unicodedata
import ast
import mc_prediction_simple_separated
import numpy as np

FEATURE_TYPE = 5
calibration = True
ISHOME = True

SIMPLE_HOME_SAVED_NETWORK_PATH = "/home/gla68/PycharmProjects/Sport-Analytic-NN/mc_prediction_simple_dir/saved_NN/mc_saved_Home_networks_feature" + str(
    FEATURE_TYPE) + "_batch16_iterate25-NEG_REWARD_GAMMA1_V3-Sequenced"
SIMPLE_AWAY_SAVED_NETWORK_PATH = "/home/gla68/PycharmProjects/Sport-Analytic-NN/mc_prediction_simple_dir/saved_NN/mc_saved_Away_networks_feature" + str(
    FEATURE_TYPE) + "_batch16_iterate25-NEG_REWARD_GAMMA1_V3-Sequenced"

calibration_store_dir = "/media/gla68/Windows/Hockey-data/mc_calibrate_all_feature_" + str(FEATURE_TYPE)

sess_nn = tf.InteractiveSession()
model_nn = mc_prediction_simple_separated.td_prediction_simple_V3()
saver = tf.train.Saver()
sess_nn.run(tf.global_variables_initializer())

if ISHOME:
    SIMPLE_SAVED_NETWORK_PATH = SIMPLE_HOME_SAVED_NETWORK_PATH
else:
    SIMPLE_SAVED_NETWORK_PATH = SIMPLE_AWAY_SAVED_NETWORK_PATH

checkpoint = tf.train.get_checkpoint_state(SIMPLE_SAVED_NETWORK_PATH)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess_nn, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    raise Exception("can't restore network")

for calibration_dir_game in os.listdir(calibration_store_dir):
    for file_name in os.listdir(calibration_store_dir + "/" + calibration_dir_game):
        if "training_data_dict_all_value" in file_name:
            calibrate_value_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
        elif "training_data_dict_all_name" in file_name:
            calibrate_name_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
        else:
            continue

    calibrate_values = (sio.loadmat(calibrate_value_name))["training_data_dict_all_value"]
    calibrate_names = (sio.loadmat(calibrate_name_name))["training_data_dict_all_name"]

    home_identifier = []
    for calibrate_name in calibrate_names:
        calibrate_name_str = unicodedata.normalize('NFKD', calibrate_name).encode('ascii', 'ignore')
        calibrate_name_dict = ast.literal_eval(calibrate_name_str)
        if calibrate_name_dict.get("home"):
            home_identifier.append(1)
        else:
            home_identifier.append(0)

    readout_t1_batch = model_nn.read_out.eval(feed_dict={model_nn.x: calibrate_values})  # get value of s

    if ISHOME:
        data_name = "model_predict_home"
    else:
        data_name = "model_predict_away"

    sio.savemat(calibration_store_dir + "/" + calibration_dir_game+"/" + "home_identifier",
                {"home_identifier": home_identifier})
    sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + data_name,
                {data_name: np.asarray(readout_t1_batch)})

    print readout_t1_batch


# def check_calibrate_home_away(calibrate_name):


# if __name__ == '__main__':
#     print "abc"
