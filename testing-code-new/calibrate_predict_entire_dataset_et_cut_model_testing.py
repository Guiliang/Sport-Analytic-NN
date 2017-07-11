import scipy.io as sio
import tensorflow as tf
import os
import unicodedata
import ast
import td_prediction_egibility_trace_cut_testing
import numpy as np

FEATURE_TYPE = 5
calibration = True
ITERATE_NUM = 75
MODEL_TYPE = "V3"
BATCH_SIZE = 8
DATA_SIZE = 100
td_prediction_egibility_trace_cut_testing.feature_num = 26
Scale = True

SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/et_dir/et_checkpoints_neg_tieC"
calibration_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Test{0}-Hockey-Training-All-feature{1}-scale-neg_reward".format(
    str(DATA_SIZE), str(FEATURE_TYPE))

sess_nn = tf.InteractiveSession()

model_nn = td_prediction_egibility_trace_cut_testing.Model(sess_nn,
                                                           td_prediction_egibility_trace_cut_testing.model_path,
                                                           td_prediction_egibility_trace_cut_testing.summary_path,
                                                           td_prediction_egibility_trace_cut_testing.checkpoint_path)

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

    readout_t1_batch = []
    for calibrate_value in calibrate_values:
        readout_value = model_nn.V.eval(feed_dict={model_nn.s_t0: [calibrate_value]})  # get value of s
        readout_t1_batch.append(readout_value[0][0])

    data_name = "model_et_cut_predict_feature_" + str(
        FEATURE_TYPE) + "_" + MODEL_TYPE + "_Iter" + str(ITERATE_NUM) + "_batch" + str(BATCH_SIZE)

    sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + "home_identifier",
                {"home_identifier": home_identifier})
    sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + data_name,
                {data_name: np.asarray(readout_t1_batch)})

    print readout_t1_batch


# def check_calibrate_home_away(calibrate_name):


# if __name__ == '__main__':
#     print "abc"
