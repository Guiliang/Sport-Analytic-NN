import ast
import os
import unicodedata

import numpy as np
import scipy.io as sio
import tensorflow as tf

import td_prediction_simple_separated

FEATURE_TYPE = 5
calibration = True
ITERATE_NUM = 75
MODEL_TYPE = "V3"

SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/saved_entire__networks_feature{0}_batch16_iterate{1}-NEG_REWARD_GAMMA1_{2}-Sequenced".format(
    str(FEATURE_TYPE), str(ITERATE_NUM), MODEL_TYPE)

calibration_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/td_calibrate_all_feature_" + str(
    FEATURE_TYPE) + "_" + MODEL_TYPE + "_Iter" + str(ITERATE_NUM)
# calibration_store_dir = "/cs/oschulte/Galen/Hockey-data/td_calibrate_all_feature_5_2017-6-01"
sess_nn = tf.InteractiveSession()

if MODEL_TYPE == "V1":
    model_nn = td_prediction_simple_separated.td_prediction_simple()
elif MODEL_TYPE == "V2":
    model_nn = td_prediction_simple_separated.td_prediction_simple_V2()
elif MODEL_TYPE == "V3":
    model_nn = td_prediction_simple_separated.td_prediction_simple_V3()
elif MODEL_TYPE == "V4":
    model_nn = td_prediction_simple_separated.td_prediction_simple_V4()
elif MODEL_TYPE == "V5":
    model_nn = td_prediction_simple_separated.td_prediction_simple_V5()
elif MODEL_TYPE == "V6":
    model_nn = td_prediction_simple_separated.td_prediction_simple_V6()
elif MODEL_TYPE == "V7":
    model_nn = td_prediction_simple_separated.td_prediction_simple_V7()
else:
    raise ValueError("Unclear model type")

saver = tf.train.Saver()
sess_nn.run(tf.global_variables_initializer())

checkpoint = tf.train.get_checkpoint_state(SIMPLE_SAVED_NETWORK_PATH)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess_nn, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    raise Exception("can't restore network: " + SIMPLE_SAVED_NETWORK_PATH)

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

    data_name = "model_predict"
    sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + "home_identifier",
                {"home_identifier": home_identifier})
    sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + data_name,
                {data_name: np.asarray(readout_t1_batch)})

    print readout_t1_batch


# def check_calibrate_home_away(calibrate_name):


# if __name__ == '__main__':
#     print "abc"