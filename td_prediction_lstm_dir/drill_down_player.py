import csv
import scipy.io as sio
import os
import unicodedata
import ast

FEATURE_TYPE = 5
calibration = True
ITERATE_NUM = 30
MODEL_TYPE = "v3"
BATCH_SIZE = 32
learning_rate = 1e-5
pre_initialize = False
MAX_TRACE_LENGTH = 10

target_player_id = 43
target_player_name = "Erik Karlsson"

if learning_rate == 1e-6:
    learning_rate_write = 6
elif learning_rate == 1e-5:
    learning_rate_write = 5
elif learning_rate == 1e-4:
    learning_rate_write = 4

data_name = "model_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}".format(
    str(FEATURE_TYPE), str(ITERATE_NUM), str(learning_rate_write), str(BATCH_SIZE), str(MAX_TRACE_LENGTH), MODEL_TYPE)

model_data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature{0}-scale-neg_reward_length-dynamic".format(
    str(FEATURE_TYPE))

csv_save_name = "./player_ranking_dir/player info name:{0} id:{1}".format(target_player_name, str(target_player_id))


def write_csv(csv_name, data_record):
    with open(csv_name, 'w') as csvfile:
        fieldnames = (data_record[0]).keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for record in data_record:
            writer.writerow(record)


def find_player_info():
    player_state_info_dict_list = []

    for calibration_dir_game in os.listdir(model_data_store_dir):
        playerIds = None
        model_data = None
        home_identifier = None
        training_data_all = None
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
            elif file_name.startswith("training_data_dict_all_name"):
                training_data_all_name = model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                training_data_all = (sio.loadmat(training_data_all_name))["training_data_dict_all_name"]
            else:
                continue

        for playerId_index in range(0, len(playerIds)):
            if target_player_id == playerIds[playerId_index]:
                player_model_home_prediction = model_data[playerId_index][0]
                player_model_away_prediction = model_data[playerId_index][1]
                player_state_info = training_data_all[playerId_index]
                player_state_info_str = unicodedata.normalize('NFKD', player_state_info).encode('ascii', 'ignore')
                player_state_info_dict = ast.literal_eval(player_state_info_str)
                home_or_away = home_identifier[playerId_index]
                player_state_info_dict.update({"model_home_prediction": player_model_home_prediction})
                player_state_info_dict.update({"model_away_prediction": player_model_away_prediction})
                if home_or_away:
                    player_team_prediction_diff = player_model_home_prediction - player_model_away_prediction
                else:
                    player_team_prediction_diff = player_model_away_prediction - player_model_home_prediction
                player_state_info_dict.update({"player_team_prediction_diff":player_team_prediction_diff})
                player_state_info_dict_list.append(player_state_info_dict)

    return player_state_info_dict_list


if __name__ == '__main__':
    player_state_info_dict_list = find_player_info()
    write_csv(csv_save_name, player_state_info_dict_list)
