import csv
import os
import scipy.io as sio

FEATURE_TYPE = 5
calibration = True
ITERATE_NUM = 30
MODEL_TYPE = "v3"
BATCH_SIZE = 32
learning_rate = 1e-5
pre_initialize = False
MAX_TRACE_LENGTH = 10

PLAYER_ID_DICT_ALL = {}
PLAYER_INTEREST = ['G', 'A', 'P', 'PlayerName']

if learning_rate == 1e-6:
    learning_rate_write = 6
elif learning_rate == 1e-5:
    learning_rate_write = 5
elif learning_rate == 1e-4:
    learning_rate_write = 4

if pre_initialize:
    pre_initialize_save = "_pre_initialize"
else:
    pre_initialize_save = ""

model_data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature{0}-scale-neg_reward_length-dynamic".format(
    str(FEATURE_TYPE))

state_model_data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/State-Hybrid-RNN-Hockey-Training-All-feature{0}-scale-neg_reward_length-dynamic".format(
    str(FEATURE_TYPE))

player_info_dir = "/cs/oschulte/Galen/PycharmProjects/Sport-Analytic-NN/td_prediction_lstm_dir/player_ranking_dir/players_2015_2016.csv"

skater_info_dir = "/cs/oschulte/Galen/PycharmProjects/Sport-Analytic-NN/td_prediction_lstm_dir/player_ranking_dir/skater_stats_2015_2016_original.csv"

data_name = "model_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}".format(
    str(FEATURE_TYPE), str(ITERATE_NUM), str(learning_rate_write), str(BATCH_SIZE), str(MAX_TRACE_LENGTH), MODEL_TYPE)

state_data_name = "model_state_cut_together_predict_Fea{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}".format(
    str(FEATURE_TYPE), str(ITERATE_NUM), str(6), str(8), str(MAX_TRACE_LENGTH), MODEL_TYPE)

def aggregate_values():
    for calibration_dir_game in os.listdir(model_data_store_dir):
        model_state_data_name = state_model_data_store_dir + "/" + calibration_dir_game + "/" + state_data_name + ".mat"
        model_state_data = (sio.loadmat(model_state_data_name))[state_data_name]
        for file_name in os.listdir(model_data_store_dir + "/" + calibration_dir_game):
            if file_name == data_name + ".mat":
                model_data_name = model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                model_data = (sio.loadmat(model_data_name))[data_name]
            elif file_name.startswith("playerId"):
                playerIds_name = model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                playerIds = (sio.loadmat(playerIds_name))["playerId"][0]
            elif file_name.startswith("home_identifier"):
                home_identifier_name = model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                home_identifier = (sio.loadmat(home_identifier_name))["home_identifier"][0]
            else:
                continue

        for player_Index in range(0, len(playerIds)):
            playerId = playerIds[player_Index]
            model_value = model_data[player_Index]
            model_state_value = model_state_data[player_Index]
            ishome = home_identifier[player_Index]
            player_value = PLAYER_ID_DICT_ALL.get(playerId)
            if player_value is None:
                if ishome:
                    PLAYER_ID_DICT_ALL.update({playerId: {"value": model_value[0] - model_value[1],
                                                          "state value": model_state_value[0] - model_state_value[1]}})
                else:
                    PLAYER_ID_DICT_ALL.update({playerId: {"value": model_value[1] - model_value[0],
                                                          "state value": model_state_value[1] - model_state_value[0]}})
            else:
                if ishome:
                    player_value_number = player_value.get("value") + model_value[0] - model_value[1]
                    player_state_value_number = player_value.get("state value") + model_state_value[0] - \
                                                model_state_value[1]
                else:
                    player_value_number = player_value.get("value") + model_value[1] - model_value[0]
                    player_state_value_number = player_value.get("state value") + model_state_value[1] - \
                                                model_state_value[0]
                PLAYER_ID_DICT_ALL.update(
                    {playerId: {"value": player_value_number, "state value": player_state_value_number}})
                # break


def transfer_save_format(playerId_skateInfo_dict):
    player_value_dict_list = []
    player_Ids = PLAYER_ID_DICT_ALL.keys()
    for player_index in range(0, len(player_Ids)):
        player_value_dict = {}
        player_Id = player_Ids[player_index]
        player_value = PLAYER_ID_DICT_ALL.get(player_Id)
        player_skateInfo = playerId_skateInfo_dict.get(player_Id)
        if player_skateInfo is not None:
            player_value_dict.update({"playerId": player_Ids[player_index]})
            player_value_dict.update(player_value)
            player_value_dict.update(player_skateInfo)
            player_value_dict_list.append(player_value_dict)
    return player_value_dict_list


def write_csv(csv_name, data_record):
    with open(csv_name, 'w') as csvfile:
        fieldnames = (data_record[0]).keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for record in data_record:
            writer.writerow(record)


def read_players_info():
    player_Info = {}
    first_row_flag = True
    with open(player_info_dir) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if first_row_flag:
                first_row_flag = False
                continue
            else:
                firstname = row[2]
                lastname = row[1]
                playerId = row[0]
                player_Info.update({firstname + " " + lastname: playerId})
    return player_Info


def combine_playerId_to_skate_info(player_Info):
    first_row_flag = True
    playerId_skateInfo_dict = {}
    row_num = 0
    with open(skater_info_dir) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=';')
        for row in read_csv:
            # row_num += 1
            if first_row_flag:
                first_row_flag = False
                item_list = row
                continue
            else:
                # print row
                player_name = row[1]
                season = row[5]
                if season == "Playoffs":
                    continue
                player_id = player_Info.get(player_name)
                if player_id is not None:
                    row_num += 1
                    item_record = {}
                    for interest_item in PLAYER_INTEREST:
                        interest_value = row[item_list.index(interest_item)]
                        item_record.update({interest_item: interest_value})
                    temp = playerId_skateInfo_dict.get(int(player_id))
                    if temp is not None:
                        print player_name
                    playerId_skateInfo_dict.update({int(player_id): item_record})
                else:
                    print player_name
    print row_num
    return playerId_skateInfo_dict


if __name__ == '__main__':
    player_Info = read_players_info()
    playerId_skateInfo_dict = combine_playerId_to_skate_info(player_Info)
    aggregate_values()
    player_value_dict_list = transfer_save_format(playerId_skateInfo_dict)
    write_csv("./player_ranking_dir/dt_lstm_player_ranking_test.csv", player_value_dict_list)
