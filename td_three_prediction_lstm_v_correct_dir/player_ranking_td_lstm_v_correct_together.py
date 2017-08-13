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
if_correct_velocity = "_v_correct_"

IS_POSIBILITY = True
IS_DIFFERENCE = False
if IS_DIFFERENCE:
    DIFFERENCE_TYPE = "skip_difference_"

PLAYER_ID_DICT_ALL = {}
PLAYER_INTEREST = ['G', 'A', 'P', 'PlayerName', 'GP', 'PlusMinus', 'PIM', 'PointPerGame', 'PPG', 'PPP', 'SHG', 'SHP',
                   'GWG', 'OTG', 'S', 'ShootingPercentage', 'TOIPerGame', 'ShiftsPerGame', 'FaceoffWinPercentage']

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

model_data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature{0}-scale" \
                       "-neg_reward{1}_length-dynamic/".format(str(FEATURE_TYPE), if_correct_velocity)

# state_model_data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/State-Hybrid-RNN-Hockey-Training-All-feature{0}-scale-neg_reward_length-dynamic".format(
#     str(FEATURE_TYPE))

player_info_dir = "./player_ranking_dir/players_2015_2016.csv"

skater_info_dir = "./player_ranking_dir/skater_stats_2015_2016_original.csv"

data_name = "model_three_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}{6}".format(
    str(FEATURE_TYPE),
    str(ITERATE_NUM),
    str(learning_rate_write),
    str(BATCH_SIZE),
    str(MAX_TRACE_LENGTH),
    str(MODEL_TYPE),
    if_correct_velocity)


# state_data_name = "model_state_cut_together_predict_Fea{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}".format(
#     str(FEATURE_TYPE), str(ITERATE_NUM), str(6), str(8), str(MAX_TRACE_LENGTH), MODEL_TYPE)


def aggregate_values():
    for calibration_dir_game in os.listdir(model_data_store_dir):
        # model_state_data_name = state_model_data_store_dir + "/" + calibration_dir_game + "/" + state_data_name + ".mat"
        # model_state_data = (sio.loadmat(model_state_data_name))[state_data_name]
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
            # model_state_value = model_state_data[player_Index]
            ishome = home_identifier[player_Index]
            player_value = PLAYER_ID_DICT_ALL.get(playerId)
            if player_value is None:
                if ishome:
                    if IS_POSIBILITY:
                        PLAYER_ID_DICT_ALL.update(
                            {playerId: {
                                "value": (model_value[0] - model_value[1]) / (
                                    model_value[0] + model_value[1] + abs(model_value[2]))}})
                        # "state value": (model_state_value[0] - model_state_value[1]) / (
                        # model_state_value[0] + model_state_value[1])}})
                    else:
                        PLAYER_ID_DICT_ALL.update({playerId: {"value": model_value[0] - model_value[1]}})
                        # "state value": model_state_value[0] - model_state_value[1]}})
                else:
                    if IS_POSIBILITY:
                        PLAYER_ID_DICT_ALL.update(
                            {playerId: {
                                "value": (model_value[1] - model_value[0]) / (
                                    model_value[0] + model_value[1] + abs(model_value[2]))}})
                        # "state value": (model_state_value[1] - model_state_value[0]) / (
                        # model_state_value[0] + model_state_value[1])}})
                    else:
                        PLAYER_ID_DICT_ALL.update({playerId: {"value": model_value[1] - model_value[0]}})
                        # "state value": model_state_value[1] - model_state_value[0]}})
            else:
                if ishome:
                    if IS_POSIBILITY:
                        player_value_number = player_value.get("value") + (model_value[0] - model_value[1]) / (
                            model_value[0] + model_value[1] + abs(model_value[2]))
                        # player_state_value_number = player_value.get("state value") + (model_state_value[0] - model_state_value[1])/(model_state_value[0] + model_state_value[1])
                    else:
                        player_value_number = player_value.get("value") + model_value[0] - model_value[1]
                        # player_state_value_number = player_value.get("state value") + model_state_value[0] - \
                        #                             model_state_value[1]
                else:
                    if IS_POSIBILITY:
                        player_value_number = player_value.get("value") + (model_value[1] - model_value[0]) / (
                            model_value[0] + model_value[1] + abs(model_value[2]))
                        # player_state_value_number = player_value.get("state value") + (model_state_value[1] - model_state_value[0])/(model_state_value[0] + model_state_value[1])
                    else:
                        player_value_number = player_value.get("value") + model_value[1] - model_value[0]
                        # player_state_value_number = player_value.get("state value") + model_state_value[1] - \
                        # model_state_value[0]
                PLAYER_ID_DICT_ALL.update(
                    {playerId: {"value": player_value_number}})
                # {playerId: {"value": player_value_number, "state value": player_state_value_number}}), "state value": player_state_value_number}})
                # break


def aggregate_diff_values():
    for calibration_dir_game in os.listdir(model_data_store_dir):
        # model_state_data_name = state_model_data_store_dir + "/" + calibration_dir_game + "/" + state_data_name + ".mat"
        # model_state_data = (sio.loadmat(model_state_data_name))[state_data_name]
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
            try:
                model_value_pre = model_data[player_Index - 1]
            except:
                model_value_pre = model_data[player_Index]
            try:
                model_value_nex = model_data[player_Index + 1]
            except:
                model_value_nex = model_data[player_Index]

            if model_value[2] < 0:
                model_value[2] = 0
            if model_value_pre[2] < 0:
                model_value_pre[2] = 0
            if model_value_nex[2] < 0:
                model_value_nex[2] = 0

            ishome = home_identifier[player_Index]
            player_value = PLAYER_ID_DICT_ALL.get(playerId)
            if player_value is None:
                if ishome:
                    if IS_POSIBILITY:
                        home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                        away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                        end_model_value = abs(model_value[2]) / (model_value[0] + model_value[1] + abs(model_value[2]))

                        home_model_value_pre = model_value_pre[0] / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                        away_model_value_pre = model_value_pre[1] / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                        end_model_value_pre = abs(model_value_pre[2]) / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                        home_model_value_nex = model_value_nex[0] / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                        away_model_value_nex = model_value_nex[1] / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                        end_model_value_nex = abs(model_value_nex[2]) / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                        if DIFFERENCE_TYPE == "back_difference_":
                            q_value = (home_model_value - home_model_value_pre) - (
                                away_model_value - away_model_value_pre)
                        elif DIFFERENCE_TYPE == "front_difference_":
                            q_value = (home_model_value_nex - home_model_value) - (
                                away_model_value_nex - away_model_value)
                        elif DIFFERENCE_TYPE == "skip_difference_":
                            q_value = (home_model_value_nex - home_model_value_pre) - (
                                away_model_value_nex - away_model_value_pre)

                        PLAYER_ID_DICT_ALL.update(
                            {playerId: {
                                "value": q_value}})
                        # "state value": (model_state_value[0] - model_state_value[1]) / (
                        # model_state_value[0] + model_state_value[1])}})
                    else:
                        home_model_value = model_value[0]
                        away_model_value = model_value[1]
                        end_model_value = abs(model_value[2])
                        home_model_value_pre = model_value_pre[0]
                        away_model_value_pre = model_value_pre[1]
                        end_model_value_pre = abs(model_value_pre[2])
                        home_model_value_nex = model_value_nex[0]
                        away_model_value_nex = model_value_nex[1]
                        end_model_value_nex = abs(model_value_nex[2])

                        if DIFFERENCE_TYPE == "back_difference_":
                            q_value = (home_model_value - home_model_value_pre) - (
                                away_model_value - away_model_value_pre)
                        elif DIFFERENCE_TYPE == "front_difference_":
                            q_value = (home_model_value_nex - home_model_value) - (
                                away_model_value_nex - away_model_value)
                        elif DIFFERENCE_TYPE == "skip_difference_":
                            q_value = (home_model_value_nex - home_model_value_pre) - (
                                away_model_value_nex - away_model_value_pre)

                        PLAYER_ID_DICT_ALL.update({playerId: {"value": q_value}})
                        # "state value": model_state_value[0] - model_state_value[1]}})
                else:
                    if IS_POSIBILITY:
                        home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                        away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                        end_model_value = abs(model_value[2]) / (model_value[0] + model_value[1] + abs(model_value[2]))

                        home_model_value_pre = model_value_pre[0] / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                        away_model_value_pre = model_value_pre[1] / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                        end_model_value_pre = abs(model_value_pre[2]) / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                        home_model_value_nex = model_value_nex[0] / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                        away_model_value_nex = model_value_nex[1] / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                        end_model_value_nex = abs(model_value_nex[2]) / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                        if DIFFERENCE_TYPE == "back_difference_":
                            q_value = (away_model_value - away_model_value_pre) - (
                                home_model_value - home_model_value_pre)
                        elif DIFFERENCE_TYPE == "front_difference_":
                            q_value = (away_model_value_nex - away_model_value) - (
                                home_model_value_nex - home_model_value)
                        elif DIFFERENCE_TYPE == "skip_difference_":
                            q_value = (away_model_value_nex - away_model_value_pre) - (
                                home_model_value_nex - home_model_value_pre
                            )

                        PLAYER_ID_DICT_ALL.update(
                            {playerId: {
                                "value": q_value}})
                    else:
                        home_model_value = model_value[0]
                        away_model_value = model_value[1]
                        end_model_value = abs(model_value[2])
                        home_model_value_pre = model_value_pre[0]
                        away_model_value_pre = model_value_pre[1]
                        end_model_value_pre = abs(model_value_pre[2])
                        home_model_value_nex = model_value_nex[0]
                        away_model_value_nex = model_value_nex[1]
                        end_model_value_nex = abs(model_value_nex[2])

                        if DIFFERENCE_TYPE == "back_difference_":
                            q_value = (away_model_value - away_model_value_pre) - (
                                home_model_value - home_model_value_pre)
                        elif DIFFERENCE_TYPE == "front_difference_":
                            q_value = (away_model_value_nex - away_model_value) - (
                                home_model_value_nex - home_model_value)
                        elif DIFFERENCE_TYPE == "skip_difference_":
                            q_value = (away_model_value_nex - away_model_value_pre) - (
                                home_model_value_nex - home_model_value_pre
                            )

                        PLAYER_ID_DICT_ALL.update(
                            {playerId: {
                                "value": q_value}})
            else:
                if ishome:
                    if IS_POSIBILITY:
                        home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                        away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                        end_model_value = abs(model_value[2]) / (model_value[0] + model_value[1] + abs(model_value[2]))

                        home_model_value_pre = model_value_pre[0] / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                        away_model_value_pre = model_value_pre[1] / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                        end_model_value_pre = abs(model_value_pre[2]) / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                        home_model_value_nex = model_value_nex[0] / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                        away_model_value_nex = model_value_nex[1] / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                        end_model_value_nex = abs(model_value_nex[2]) / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                        if DIFFERENCE_TYPE == "back_difference_":
                            q_value = (home_model_value - home_model_value_pre) - (
                                away_model_value - away_model_value_pre)
                        elif DIFFERENCE_TYPE == "front_difference_":
                            q_value = (home_model_value_nex - home_model_value) - (
                                away_model_value_nex - away_model_value)
                        elif DIFFERENCE_TYPE == "skip_difference_":
                            q_value = (home_model_value_nex - home_model_value_pre) - (
                                away_model_value_nex - away_model_value_pre)

                        player_value_number = player_value.get("value") + q_value
                    else:
                        home_model_value = model_value[0]
                        away_model_value = model_value[1]
                        end_model_value = abs(model_value[2])
                        home_model_value_pre = model_value_pre[0]
                        away_model_value_pre = model_value_pre[1]
                        end_model_value_pre = abs(model_value_pre[2])
                        home_model_value_nex = model_value_nex[0]
                        away_model_value_nex = model_value_nex[1]
                        end_model_value_nex = abs(model_value_nex[2])

                        if DIFFERENCE_TYPE == "back_difference_":
                            q_value = (home_model_value - home_model_value_pre) - (
                                away_model_value - away_model_value_pre)
                        elif DIFFERENCE_TYPE == "front_difference_":
                            q_value = (home_model_value_nex - home_model_value) - (
                                away_model_value_nex - away_model_value)
                        elif DIFFERENCE_TYPE == "skip_difference_":
                            q_value = (home_model_value_nex - home_model_value_pre) - (
                                away_model_value_nex - away_model_value_pre)

                        player_value_number = player_value.get("value") + q_value

                else:
                    if IS_POSIBILITY:
                        home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                        away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                        end_model_value = abs(model_value[2]) / (model_value[0] + model_value[1] + abs(model_value[2]))

                        home_model_value_pre = model_value_pre[0] / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                        away_model_value_pre = model_value_pre[1] / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                        end_model_value_pre = abs(model_value_pre[2]) / (
                            model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                        home_model_value_nex = model_value_nex[0] / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                        away_model_value_nex = model_value_nex[1] / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                        end_model_value_nex = abs(model_value_nex[2]) / (
                            model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                        if DIFFERENCE_TYPE == "back_difference_":
                            q_value = (away_model_value - away_model_value_pre) - (
                                home_model_value - home_model_value_pre)
                        elif DIFFERENCE_TYPE == "front_difference_":
                            q_value = (away_model_value_nex - away_model_value) - (
                                home_model_value_nex - home_model_value)
                        elif DIFFERENCE_TYPE == "skip_difference_":
                            q_value = (away_model_value_nex - away_model_value_pre) - (
                                home_model_value_nex - home_model_value_pre
                            )
                        player_value_number = player_value.get("value") + q_value


                    else:
                        home_model_value = model_value[0]
                        away_model_value = model_value[1]
                        end_model_value = abs(model_value[2])
                        home_model_value_pre = model_value_pre[0]
                        away_model_value_pre = model_value_pre[1]
                        end_model_value_pre = abs(model_value_pre[2])
                        home_model_value_nex = model_value_nex[0]
                        away_model_value_nex = model_value_nex[1]
                        end_model_value_nex = abs(model_value_nex[2])

                        if DIFFERENCE_TYPE == "back_difference_":
                            q_value = (away_model_value - away_model_value_pre) - (
                                home_model_value - home_model_value_pre)
                        elif DIFFERENCE_TYPE == "front_difference_":
                            q_value = (away_model_value_nex - away_model_value) - (
                                home_model_value_nex - home_model_value)
                        elif DIFFERENCE_TYPE == "skip_difference_":
                            q_value = (away_model_value_nex - away_model_value_pre) - (
                                home_model_value_nex - home_model_value_pre
                            )
                        player_value_number = player_value.get("value") + q_value

                PLAYER_ID_DICT_ALL.update(
                    {playerId: {"value": player_value_number}})
                # {playerId: {"value": player_value_number, "state value": player_state_value_number}}), "state value": player_state_value_number}})
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
    if IS_DIFFERENCE:
        aggregate_diff_values()
        player_value_dict_list = transfer_save_format(playerId_skateInfo_dict)
    else:
        aggregate_values()
        player_value_dict_list = transfer_save_format(playerId_skateInfo_dict)
    if IS_POSIBILITY:
        possi_write = "_possibility"
    else:
        possi_write = ""
    if IS_DIFFERENCE:
        diff_write = DIFFERENCE_TYPE
    else:
        diff_write = ""

    write_csv("./player_ranking_dir/dt{0}_{1}lstm_player_ranking_test.csv".format(possi_write, diff_write),
              player_value_dict_list)
