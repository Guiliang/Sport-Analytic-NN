import scipy.io as sio
import tensorflow as tf
import os
import unicodedata
import ast
import td_prediction_lstm_cut_together
import numpy as np

FEATURE_TYPE = 5
calibration = True
ITERATE_NUM = 50
BATCH_SIZE = 8
MAX_LENGTH = 10
TEST_LENGTH = 100
MODEL_TYPE = "v3"

SIMPLE_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/hybrid_sl_saved_NN/Scale-Test{0}-cut_together_saved_networks_feature{1}_batch{2}_iterate{3}_{4}".format(
    str(TEST_LENGTH), str(FEATURE_TYPE), str(BATCH_SIZE), str(ITERATE_NUM), str(MODEL_TYPE))

calibration_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Test{0}-Hybrid-RNN-Hockey-Training-All-feature{1}-scale" \
                        "-neg_reward_length-dynamic/".format(str(TEST_LENGTH), str(FEATURE_TYPE))

data_name = "model_cut_together_predict_Feature{0}_Iter{1}_Batch{2}_MaxLength{3}_Type{4}".format(str(FEATURE_TYPE),
                                                                                                 str(ITERATE_NUM),
                                                                                                 str(BATCH_SIZE),
                                                                                                 str(MAX_LENGTH),
                                                                                                 str(MODEL_TYPE))

# "/cs/oschulte/Galen/models/hybrid_sl_saved_NN/Scale-Test100-cut_together_saved_networks_feature5_batch8_iterate50_v3"


def generate_calibration_data():
    sess_nn = tf.InteractiveSession()
    # if MODEL_TYPE == "V1":
    #     model_nn = td_prediction_lstm_cut_together_testing.td_prediction_lstm()
    # elif MODEL_TYPE == "V2":
    #     model_nn = td_prediction_lstm_cut_together_testing.td_prediction_lstm_V2()
    if MODEL_TYPE == "v3":
        model_nn = td_prediction_lstm_cut_together.td_prediction_lstm_V3()
    else:
        raise ValueError("MODEL_TYPE error")
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
            if "hybrid_input_state" in file_name:
                calibrate_value_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            # elif "training_data_dict_all_name" in file_name:
            #     calibrate_name_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            elif "hybrid_trace_length" in file_name:
                calibrate_trace_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            else:
                continue

        calibrate_values = (sio.loadmat(calibrate_value_name))["hybrid_input_state"]
        # calibrate_names = (sio.loadmat(calibrate_name_name))["training_data_dict_all_name"]
        calibration_trace_length = ((sio.loadmat(calibrate_trace_name))["hybrid_trace_length"])[0]
        calibration_trace_length = handle_trace_length(calibration_trace_length)

        for trace_length_index in range(0, len(calibration_trace_length)):
            if calibration_trace_length[trace_length_index] > 10:
                calibration_trace_length[trace_length_index] = 10

        # home_identifier = []
        # for calibrate_name in calibrate_names:
        #     calibrate_name_str = unicodedata.normalize('NFKD', calibrate_name).encode('ascii', 'ignore')
        #     calibrate_name_dict = ast.literal_eval(calibrate_name_str)
        #     if calibrate_name_dict.get("home"):
        #         home_identifier.append(1)
        #     else:
        #         home_identifier.append(0)

        readout_t1_batch = model_nn.read_out.eval(feed_dict={model_nn.rnn_input: calibrate_values,
                                                             model_nn.trace_lengths: calibration_trace_length})  # get value of s

        # sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + "home_identifier",
        #             {"home_identifier": home_identifier})
        sio.savemat(calibration_store_dir + "/" + calibration_dir_game + "/" + data_name,
                    {data_name: np.asarray(readout_t1_batch)})

        print readout_t1_batch


def handle_trace_length(state_trace_length):
    """
    transform format of trace length
    :return:
    """
    trace_length_record = []
    for length in state_trace_length:
        for sub_length in range(0, length):
            trace_length_record.append(sub_length + 1)
    return trace_length_record


if __name__ == '__main__':
    generate_calibration_data()
