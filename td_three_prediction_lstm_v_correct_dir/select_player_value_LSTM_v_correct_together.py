import ast
import csv
import os
import scipy.io as sio
import unicodedata

TARGET_PLAYER_ID = int(2)
PLAYER_NAME = "Taylor Hall"
COMPUTE_IMPACT = True

FEATURE_TYPE = 5
calibration = True
ITERATE_NUM = 50
MODEL_TYPE = "v3"
BATCH_SIZE = 32
learning_rate = 1e-4
pre_initialize = False
MAX_TRACE_LENGTH = 10
if_correct_velocity = "_v_correct_"
ROUND_NUMBER = 65

model_data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature{0}-scale" \
                       "-neg_reward{1}_length-dynamic".format(str(FEATURE_TYPE), if_correct_velocity)

if learning_rate == 1e-6:
    learning_rate_write = 6
elif learning_rate == 1e-5:
    learning_rate_write = 5
elif learning_rate == 1e-4:
    learning_rate_write = 4

data_name = "model_three_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}{6}".format(
    str(FEATURE_TYPE),
    str(ITERATE_NUM),
    str(learning_rate_write),
    str(BATCH_SIZE),
    str(MAX_TRACE_LENGTH),
    str(MODEL_TYPE),
    if_correct_velocity)


def find_player_values():
    player_value_record = []
    player_impact_record = []
    player_impact_input_record = []
    player_input_record = []
    for calibration_dir_game in os.listdir(model_data_store_dir):
        print calibration_dir_game
        # model_state_data_name = state_model_data_store_dir + "/" + calibration_dir_game + "/" + state_data_name + ".mat"
        # model_state_data = (sio.loadmat(model_state_data_name))[state_data_name]
        for file_name in os.listdir(model_data_store_dir + "/" + calibration_dir_game):
            if file_name == data_name + ".mat":
                model_data_name = model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                model_data = (sio.loadmat(model_data_name))[data_name]
            elif "dynamic_rnn_input" in file_name:
                state_input_name = file_name
                state_input = sio.loadmat(model_data_store_dir + "/" + calibration_dir_game + "/" + state_input_name)
                state_input = (state_input['dynamic_feature_input'])
            elif file_name.startswith("playerId"):
                playerIds_name = model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                playerIds = (sio.loadmat(playerIds_name))["playerId"][0]
            elif file_name.startswith("home_identifier"):
                home_identifier_name = model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                home_identifier = (sio.loadmat(home_identifier_name))["home_identifier"][0]
            elif "training_data_dict_all_name" in file_name:
                training_data_dict_all_name = model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                training_data_dict_all = ((sio.loadmat(training_data_dict_all_name))["training_data_dict_all_name"])
            else:
                continue

        for player_Index in range(0, len(playerIds)):
            playerId = playerIds[player_Index]

            if playerId == TARGET_PLAYER_ID:
                model_value = model_data[player_Index]
                home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                end_model_value = abs(model_value[2]) / (model_value[0] + model_value[1] + abs(model_value[2]))

                state_input_value = state_input[player_Index]
                state_input_value_list = []
                for state in state_input_value:
                    if check_if_zero_empty(state.tolist()):
                        state_input_value_list = state.tolist() + state_input_value_list
                    else:
                        state_input_value_list = state_input_value_list + state.tolist()
                home_or_away = home_identifier[player_Index]
                player_input_record.append(state_input_value_list)
                if home_or_away:
                    player_value_record.append([home_model_value])
                else:
                    player_value_record.append([away_model_value])

                if COMPUTE_IMPACT:
                    if player_Index - 1 >= 0:
                        training_data_dict_all_pre = training_data_dict_all[player_Index - 1]
                        training_data_dict_all_pre_str = unicodedata.normalize('NFKD',
                                                                               training_data_dict_all_pre).encode(
                            'ascii', 'ignore')
                        training_data_dict_all_pre_dict = ast.literal_eval(training_data_dict_all_pre_str)

                        if training_data_dict_all_pre_dict.get('action') == "goal":
                            model_value_pre = model_data[player_Index]
                            state_input_value_pre = state_input[player_Index]
                        else:
                            model_value_pre = model_data[player_Index - 1]
                            state_input_value_pre = state_input[player_Index - 1]
                    else:
                        model_value_pre = model_data[player_Index]
                        state_input_value_pre = state_input[player_Index]

                    home_model_value_pre = model_value_pre[0] / (
                        model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                    away_model_value_pre = model_value_pre[1] / (
                        model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                    end_model_value_pre = abs(model_value_pre[2]) / (
                        model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                    if home_or_away:
                        player_impact_record.append([home_model_value - home_model_value_pre])
                    else:
                        player_impact_record.append([away_model_value - away_model_value_pre])

                    state_input_value_pre_list = []
                    for state_pre in state_input_value_pre:
                        if check_if_zero_empty(state_pre.tolist()):
                            state_input_value_pre_list = state_pre.tolist() + state_input_value_pre_list
                        else:
                            state_input_value_pre_list = state_input_value_pre_list + state_pre.tolist()
                    player_impact_input_record.append(state_input_value_pre_list + state_input_value_list)
    if COMPUTE_IMPACT:
        return player_value_record, player_input_record, player_impact_record, player_impact_input_record
    else:
        return player_value_record, player_input_record, None, None


def write_csv(csv_name, data_record):
    with open(csv_name, 'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        for data_entry in data_record:
            wr.writerow(data_entry)


def check_if_zero_empty(check_list):
    is_zero_flag = True
    for check_value in check_list:
        if check_value != float(0):
            is_zero_flag = False

    return is_zero_flag


if __name__ == '__main__':
    csv_value_name = "./decision-tree/sequence-value-{0}-2018-08-28.csv".format(PLAYER_NAME)
    csv_input_name = "./decision-tree/sequence-input-{0}-2018-08-28.csv".format(PLAYER_NAME)
    csv_impact_value_name = "./decision-tree/sequence-impact-value-{0}-2018-08-28.csv".format(PLAYER_NAME)
    csv_impact_input_name = "./decision-tree/sequence-impact-input-{0}-2018-08-28.csv".format(PLAYER_NAME)
    value_record, input_record, impact_value_record, impact_input_record = find_player_values()
    write_csv(csv_value_name, value_record)
    write_csv(csv_input_name, input_record)
    write_csv(csv_impact_value_name, impact_value_record)
    write_csv(csv_impact_input_name, impact_input_record)
