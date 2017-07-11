import ast

import scipy.io as sio
import tensorflow as tf

import td_prediction_simple_dir.old_code.td_prediction_simple_separated

FEATURE_TYPE = 5
MOTION_TYPE = "lpr"
calibration = True
ISHOME = True

SIMPLE_HOME_SAVED_NETWORK_PATH = "/home/gla68/PycharmProjects/Sport-Analytic-NN/td_prediction_simple_dir/saved_NN/saved_Home_networks_feature"+str(FEATURE_TYPE)+"_batch16_iterate2Converge-NEG_REWARD_GAMMA1_V3-Sequenced"
SIMPLE_AWAY_SAVED_NETWORK_PATH = "/home/gla68/PycharmProjects/Sport-Analytic-NN/td_prediction_simple_dir/saved_NN/saved_Away_networks_feature"+str(FEATURE_TYPE)+"_batch16_iterate2Converge-NEG_REWARD_GAMMA1_V3-Sequenced"

if calibration:
    # SIMULATION_DATA_PATH = ""
    calibration_store_dir = "/media/gla68/Windows/Hockey-data/Simulation-data-feature" + str(
        FEATURE_TYPE) + "/calibration/"
    calibration_home_store_file_name = "calibration-"+MOTION_TYPE+"-feature" + str(FEATURE_TYPE) + "-"
    calibration_away_store_file_name = "Away_calibration-"+MOTION_TYPE+"-feature" + str(FEATURE_TYPE) + "-"
    # SIMULATION_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature5/calibration/calibration-lpr-feature5-['xAdjCoord', 'scoreDifferential', 'velocity_x', 'away', 'time remained', 'Penalty', 'yAdjCoord', 'velocity_y', 'duration', 'home'][-32, 0, -1, 0, 3594, 0, -1, 0, 2, 1].mat"
    # SIMULATION_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature5/calibration/Away_calibration-faceoff-feature5-['Penalty', 'xAdjCoord', 'scoreDifferential', 'yAdjCoord', 'velocity_x', 'velocity_y', 'duration', 'home', 'away', 'time remained'][0, -34, 0, 1, -2, 1, 2, 0, 1, 3595].mat"
    # SIMULATION_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature5/calibration/calibration-faceoff-feature5-['Penalty', 'xAdjCoord', 'scoreDifferential', 'yAdjCoord', 'velocity_x', 'velocity_y', 'duration', 'home', 'away', 'time remained'][0, -33, 0, -1, -1, -1, 2, 1, 0, 3595].mat"
    # SIMULATION_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature5/calibration/calibration-lpr-feature5-['Penalty', 'xAdjCoord', 'scoreDifferential', 'yAdjCoord', 'velocity_x', 'velocity_y', 'duration', 'home', 'away', 'time remained'][0, -29, 0, -2, 0, 0, 9, 1, 0, 2198].mat"
    # SIMULATION_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature5/calibration/Away_calibration-lpr-feature5-['Penalty', 'xAdjCoord', 'scoreDifferential', 'yAdjCoord', 'velocity_x', 'velocity_y', 'duration', 'home', 'away', 'time remained'][0, -30.5, 0, 3.41, 0, 0, 5, 0, 1, 2201].mat"
    # SIMULATION_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature5/calibration/calibration-lpr-feature5-['Penalty', 'xAdjCoord', 'scoreDifferential', 'yAdjCoord', 'velocity_x', 'velocity_y', 'duration', 'home', 'away', 'time remained'][0, -31, 0, 0, 0, 0, 3, 1, 0, 903].mat"
    # SIMULATION_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature5/calibration/Away_calibration-lpr-feature5-['Penalty', 'xAdjCoord', 'scoreDifferential', 'yAdjCoord', 'velocity_x', 'velocity_y', 'duration', 'home', 'away', 'time remained'][0, -28, 0, -2, 0, 0, 3, 0, 1, 902].mat"
    # SIMULATION_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature8/calibration/calibration-shot-feature8-['scoreDifferential', 'period', 'Penalty', 'velocity_x', 'velocity_y', 'duration'][0, 3, 0, 0, 0, 0].mat"
    # if ISHOME:
    #     SIMPLE_SAVED_NETWORK_PATH = SIMPLE_HOME_SAVED_NETWORK_PATH
    # else:
    #     SIMPLE_SAVED_NETWORK_PATH = SIMPLE_AWAY_SAVED_NETWORK_PATH
else:
    ACTION_TYPE = "shot"
    STIMULATE_TYPE = "position"
    if ISHOME:
        SIMULATION_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature" + str(
            FEATURE_TYPE) + "/" + STIMULATE_TYPE + "_simulation/" + STIMULATE_TYPE + "_simulation-" + ACTION_TYPE + "-feature" + str(
            FEATURE_TYPE) + "-[][].mat"
        SIMPLE_SAVED_NETWORK_PATH = SIMPLE_HOME_SAVED_NETWORK_PATH
    else:
        SIMULATION_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature" + str(
            FEATURE_TYPE) + "/" + STIMULATE_TYPE + "_simulation/Away_" + STIMULATE_TYPE + "_simulation-" + ACTION_TYPE + "-feature" + str(
            FEATURE_TYPE) + "-[][].mat"
        SIMPLE_SAVED_NETWORK_PATH = SIMPLE_AWAY_SAVED_NETWORK_PATH

    RNN_SAVED_NETWORK_PATH = "./saved_NN/"


def nn_simulation():
    sess_nn = tf.InteractiveSession()
    model_nn = td_prediction_simple_dir.old_code.td_prediction_simple_separated.td_prediction_simple_V3()

    simulate_data = sio.loadmat(SIMULATION_DATA_PATH)
    simulate_data = (simulate_data['simulate_data'])

    saver = tf.train.Saver()
    sess_nn.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(SIMPLE_SAVED_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess_nn, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        raise Exception("can't restore network")

    readout_t1_batch = model_nn.read_out.eval(feed_dict={model_nn.x: simulate_data})
    return (readout_t1_batch.tolist())[0]
    # print (readout_t1_batch.tolist())

    # draw_value_over_position(readout_t1_batch)
    #
    # print(max(readout_t1_batch))
    # print(min(readout_t1_batch))
    #
    # return readout_t1_batch


def check_calibrate_target(specify_dict):
    home_id = specify_dict["home"]
    if home_id:
        home_away = "Home"
    else:
        home_away = "Away"
    penalty = specify_dict["Penalty"]
    time_remained = specify_dict["time remained"]
    scoreDifferential = specify_dict["scoreDifferential"]
    cali_id = home_away + "-" + str(penalty) + "-" + str(time_remained) + "-" + str(scoreDifferential)
    return cali_id


def calibrate_all():
    sess_nn = tf.InteractiveSession()
    model_nn = td_prediction_simple_dir.old_code.td_prediction_simple_separated.td_prediction_simple_V3()
    with open('dict_specify_record', 'r') as file:
        specify = file.readlines()
        for specify_line in specify:
            specify_dict = ast.literal_eval(specify_line)
            ISHOME = int(specify_dict["home"])
            if ISHOME:
                SIMPLE_SAVED_NETWORK_PATH = SIMPLE_HOME_SAVED_NETWORK_PATH
                calibrate_dir = calibration_store_dir + calibration_home_store_file_name + str(
                    specify_dict.keys()) + str(
                    specify_dict.values())
            else:
                SIMPLE_SAVED_NETWORK_PATH = SIMPLE_AWAY_SAVED_NETWORK_PATH
                calibrate_dir = calibration_store_dir + calibration_away_store_file_name + str(
                    specify_dict.keys()) + str(
                    specify_dict.values())
            cali_id = check_calibrate_target(specify_dict)
            SIMULATION_DATA_PATH = calibrate_dir

            simulate_data = sio.loadmat(SIMULATION_DATA_PATH)
            simulate_data = (simulate_data['simulate_data'])

            saver = tf.train.Saver()
            sess_nn.run(tf.global_variables_initializer())

            checkpoint = tf.train.get_checkpoint_state(SIMPLE_SAVED_NETWORK_PATH)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess_nn, checkpoint.model_checkpoint_path)
                # print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                raise Exception("can't restore network")

            readout_t1_batch = model_nn.read_out.eval(feed_dict={model_nn.x: simulate_data})
            calibrate_nn_value = ((readout_t1_batch.tolist())[0])[0]

            print cali_id + ": " + str(calibrate_nn_value)


# def draw_value_over_position(y):
#     max_data = y.max()
#     min_data = y.min()
#     # max_data = (max(y))[0]
#     # min_data = (min(y))[0]
#     scale = float(80) / (max_data - min_data)
#     y_list = y.tolist()
#     y_deal = []
#     y_deal
#     for y_data in y_list:
#         # y_data_deal = ((y_data[0] - min_data) * scale) - 40
#         y_deal.append(y_data[0])
#
#     if STIMULATE_TYPE == "angel":
#         x = np.arange(-0, 360, float(360) / 120)
#     elif STIMULATE_TYPE == "position":
#         x = np.arange(-100, 100, 2)
#     plt.plot(x, y_deal)
#     # img = imread('./hockey-field.png')
#     # plt.imshow(img, extent=[-100, 100, -50, 50])
#     plt.show()
#     return None


if __name__ == '__main__':
    calibrate_all()
