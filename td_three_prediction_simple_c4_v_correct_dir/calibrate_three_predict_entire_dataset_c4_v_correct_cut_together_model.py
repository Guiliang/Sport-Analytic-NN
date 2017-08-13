import scipy.io as sio
import tensorflow as tf
import os
import unicodedata
import ast
import td_three_prediction_simple_c4_v_correct_cut_together
import numpy as np

FEATURE_TYPE = 5
calibration = True
ITERATE_NUM = 50
MODEL_TYPE = "V3"
BATCH_SIZE = 32
td_three_prediction_simple_c4_v_correct_cut_together.feature_num = 26 * 4
pre_initialize = False
if_correct_velocity = "_v_correct_"
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

SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/Scale-three-c4-cut_saved_entire_together_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(BATCH_SIZE) + "_iterate" + str(ITERATE_NUM) + "_lr" + str(
    learning_rate) + "-NEG_REWARD_GAMMA1_" + MODEL_TYPE + "-Sequenced" + pre_initialize_situation + if_correct_velocity

calibration_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/c4-Hockey-Training-All-feature{0}-scale-neg_reward{1}".format(str(FEATURE_TYPE), if_correct_velocity)


"/cs/oschulte/Galen/models/saved_NN/Scale-three-c4-cut_saved_entire_together_networks_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced_v_correct_"


data_name = "model_c4_three_cut_together_predict_feature{0}_{4}_Iter{1}_lr{2}_batch{3}{5}".format(
    str(FEATURE_TYPE),
    str(ITERATE_NUM),
    str(learning_rate_write),
    str(BATCH_SIZE),
    str(MODEL_TYPE),
    if_correct_velocity)

sess_nn = tf.InteractiveSession()

# if MODEL_TYPE == "V1":
#     model_nn = td_prediction_simple_cut_together_testing.td_prediction_simple()
# elif MODEL_TYPE == "V2":
#     Scale-three-cut_saved_entire_together_networks_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequencedmodel_nn = td_prediction_simple_cut_together_testing.td_prediction_simple_V2()
if MODEL_TYPE == "V3":
    model_nn = td_three_prediction_simple_c4_v_correct_cut_together.td_prediction_simple_V3()
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


def generate_calibration_data():
    for calibration_dir_game in os.listdir(calibration_store_dir):
        calibrate_value_name = ""
        calibrate_name_name = ""
        for file_name in os.listdir(calibration_store_dir + "/" + calibration_dir_game):
            if "state_c4" in file_name:
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

        sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + "home_identifier",
                    {"home_identifier": home_identifier})
        sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + data_name,
                    {data_name: np.asarray(readout_t1_batch)})

        print readout_t1_batch


def handle_trace_length(state_trace_length):
    """
    transform format of trace length
    :return:
    """
    trace_length_record = []
    try:
        for length in state_trace_length:
            for sub_length in range(0, int(length)):
                trace_length_record.append(sub_length + 1)
    except:
        print "error"
    return trace_length_record

# def check_calibrate_home_away(calibrate_name):


if __name__ == '__main__':
    generate_calibration_data()
#     print "abc"
