import ast
import os
import unicodedata

import numpy as np
import scipy.io as sio
import tensorflow as tf

import td_prediction_lstm_cut_separate_testing

FEATURE_TYPE = 5
calibration = True
ISHOME = False
ITERATE_NUM = 25
MODEL_TYPE = "V3"
BATCH_SIZE = 8
DATA_SIZE = 100

SIMPLE_HOME_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/Test{0}-cut_home_saved_networks_feature{1}_batch{2}_iterate{3}".format(
    str(DATA_SIZE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM))
SIMPLE_AWAY_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/Test{0}-cut_away_saved_networks_feature{1}_batch{2}_iterate{3}".format(
    str(DATA_SIZE), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM))

calibration_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Test{0}-Hybrid-RNN-Hockey-Training-All-feature{1}-scale-neg_reward_length-dynamic".format(str(DATA_SIZE), str(FEATURE_TYPE))
sess_nn = tf.InteractiveSession()

if MODEL_TYPE == "V1":
    model_nn = td_prediction_lstm_cut_separate_testing.td_prediction_lstm()
elif MODEL_TYPE == "V2":
    model_nn = td_prediction_lstm_cut_separate_testing.td_prediction_lstm_v2()
else:
    raise ValueError("Unclear model type")

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
    raise Exception("can't restore network: " + SIMPLE_SAVED_NETWORK_PATH)

for calibration_dir_game in os.listdir(calibration_store_dir):
    for file_name in os.listdir(calibration_store_dir + "/" + calibration_dir_game):
        if "state" in file_name:
            calibrate_value_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
        elif "training_data_dict_all_name" in file_name:
            calibrate_name_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
        else:
            continue

    calibrate_values = (sio.loadmat(calibrate_value_name))["state"]
    calibrate_names = (sio.loadmat(calibrate_name_name))["training_data_dict_all_name"]

    home_identifier = []
    for calibrate_name in calibrate_names:
        calibrate_name_str = unicodedata.normalize('NFKD', calibrate_name).encode('ascii', 'ignore')
        calibrate_name_dict = ast.literal_eval(calibrate_name_str)
        if calibrate_name_dict.get("home"):
            home_identifier.append(1)
        else:
            home_identifier.append(0)

    # readout_t1_batch = model_nn.read_out.eval(feed_dict={model_nn.x: calibrate_values})  # get value of s #TODO: where is trace length?

    if ISHOME:
        data_name = "lstm_model_cut_predict_home_feature_" + str(
            FEATURE_TYPE) + "_" + MODEL_TYPE + "_Iter" + str(ITERATE_NUM) + "_batch" + str(BATCH_SIZE)
    else:
        data_name = "lstm_model_cut_predict_away_feature_" + str(
            FEATURE_TYPE) + "_" + MODEL_TYPE + "_Iter" + str(ITERATE_NUM) + "_batch" + str(BATCH_SIZE)

    sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + "home_identifier",
                {"home_identifier": home_identifier})
    sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + data_name,
                {data_name: np.asarray(readout_t1_batch)})

    print readout_t1_batch


# def check_calibrate_home_away(calibrate_name):


# if __name__ == '__main__':
#     print "abc"
