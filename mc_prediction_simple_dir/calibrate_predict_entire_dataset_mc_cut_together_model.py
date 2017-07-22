import scipy.io as sio
import tensorflow as tf
import os
import unicodedata
import ast
import mc_together_prediction_simple_cut
import numpy as np

FEATURE_TYPE = 5
calibration = True
Scale = True
ITERATE_NUM = 50
MODEL_TYPE = "V3"
BATCH_SIZE = 32
mc_together_prediction_simple_cut.feature_num = 26
pre_initialize = True
learning_rate = 1e-5
if pre_initialize:
    pre_initialize_situation = "-pre_initialize"
    pre_initialize_save = "_pre_initialize"
else:
    pre_initialize_situation = ""
    pre_initialize_save = ""

if Scale:
    SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/mc-Scale-cut_saved_entire_together_networks_feature{0}_batch{1}_iterate{2}_lr{3}-NEG_REWARD_GAMMA1_{4}-Sequenced{5}".format(
        str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate), MODEL_TYPE, pre_initialize_situation)
    calibration_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Training-All-feature{0}-scale-neg_reward".format(
        str(FEATURE_TYPE))
else:
    SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/mc-cut_saved_entire_together_networks_feature{0}_batch{1}_iterate{2}-NEG_REWARD_GAMMA1_{3}-Sequenced{4}".format(
        str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(learning_rate), MODEL_TYPE, pre_initialize_situation)
    calibration_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Training-All-feature{0}-neg_reward".format(
        str(FEATURE_TYPE))

sess_nn = tf.InteractiveSession()

# if MODEL_TYPE == "V1":
#     model_nn = td_prediction_simple_cut_together_testing.td_prediction_simple()
# elif MODEL_TYPE == "V2":
#     model_nn = td_prediction_simple_cut_together_testing.td_prediction_simple_V2()
if MODEL_TYPE == "V3":
    model_nn = mc_together_prediction_simple_cut.td_prediction_simple_V3()
# elif MODEL_TYPE == "V4":
#     model_nn = td_prediction_simple_cut_together_testing.td_prediction_simple_V4()
# elif MODEL_TYPE == "V5":
#     model_nn = td_prediction_simple_cut_together_testing.td_prediction_simple_V5()
# elif MODEL_TYPE == "V6":
#     model_nn = td_prediction_simple_cut_together_testing.td_prediction_simple_V6()
# elif MODEL_TYPE == "V7":
#     model_nn = td_prediction_simple_cut_together_testing.td_prediction_simple_V7()
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
    calibrate_value_name = ""
    calibrate_name_name = ""
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

    readout_t1_batch = model_nn.read_out.eval(feed_dict={model_nn.x: calibrate_values})  # get value of s

    data_name = "model_mc_cut_together_predict_feature_" + str(
        FEATURE_TYPE) + "_" + MODEL_TYPE + "_Iter" + str(ITERATE_NUM) + "_lr" + str(learning_rate) + "_batch" + str(
        BATCH_SIZE) + pre_initialize_save

    sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + "home_identifier",
                {"home_identifier": home_identifier})
    sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + data_name,
                {data_name: np.asarray(readout_t1_batch)})

    print readout_t1_batch


# def check_calibrate_home_away(calibrate_name):


# if __name__ == '__main__':
#     print "abc"
