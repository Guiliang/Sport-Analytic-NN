import os
import scipy.io as sio
import unicodedata


class PlayerImpact:

    def __init__(self):
        self.player_id_dict_all_by_match={}
        self.PLAYER_ID_DICT_ALL = {}
        self.DIFFERENCE_TYPE = None
        self.data_name = None
        self.model_data_store_dir = None
        pass

    def aggregate_match_diff_values(self, calibration_dir_game, teamId_target):
        for file_name in os.listdir(self.model_data_store_dir + "/" + calibration_dir_game):
            if file_name == self.data_name + ".mat":
                model_data_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                model_data = (sio.loadmat(model_data_name))[self.data_name]
            elif file_name.startswith("playerId"):
                playerIds_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                playerIds = (sio.loadmat(playerIds_name))["playerId"][0]
            elif file_name.startswith("teamId"):
                teamIds_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                teamIds = (sio.loadmat(teamIds_name))["teamId"][0]
            elif file_name.startswith("home_identifier"):
                home_identifier_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                home_identifier = (sio.loadmat(home_identifier_name))["home_identifier"][0]
            elif "training_data_dict_all_name" in file_name:
                training_data_dict_all_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                training_data_dict_all = ((sio.loadmat(training_data_dict_all_name))["training_data_dict_all_name"])
            else:
                continue

        for player_Index in range(0, len(playerIds)):
            playerId = playerIds[player_Index]
            teamId = teamIds[player_Index]
            if int(teamId_target) == int(teamId):
                model_value = model_data[player_Index]
                if player_Index - 1 >= 0:  # define model pre
                    training_data_dict_all_pre = training_data_dict_all[player_Index - 1]
                    training_data_dict_all_pre_str = unicodedata.normalize('NFKD', training_data_dict_all_pre).encode(
                        'ascii', 'ignore')
                    training_data_dict_all_pre_dict = ast.literal_eval(training_data_dict_all_pre_str)

                    if training_data_dict_all_pre_dict.get('action') == "goal":
                        model_value_pre = model_data[player_Index]  # the goal cancel out here, just as we cut the game
                    else:
                        model_value_pre = model_data[player_Index - 1]
                else:
                    model_value_pre = model_data[player_Index]

                if player_Index + 1 <= len(playerIds):  # define model next
                    training_data_dict_all_nex = training_data_dict_all[player_Index]
                    training_data_dict_all_nex_str = unicodedata.normalize('NFKD', training_data_dict_all_nex).encode(
                        'ascii', 'ignore')
                    training_data_dict_all_nex_dict = ast.literal_eval(training_data_dict_all_nex_str)

                    if training_data_dict_all_nex_dict.get('action') == "goal":
                        model_value_nex = model_data[player_Index]
                    else:
                        model_value_nex = model_data[player_Index - 1]

                else:
                    model_value_nex = model_data[player_Index]

                ishome = home_identifier[player_Index]
                player_value = self.player_id_dict_all_by_match.get(playerId)

                home_model_value = model_value[0]
                away_model_value = model_value[1]
                end_model_value = abs(model_value[2])
                home_model_value_pre = model_value_pre[0]
                away_model_value_pre = model_value_pre[1]
                end_model_value_pre = abs(model_value_pre[2])
                home_model_value_nex = model_value_nex[0]
                away_model_value_nex = model_value_nex[1]
                end_model_value_nex = abs(model_value_nex[2])

                if ishome:
                    if self.DIFFERENCE_TYPE == "back_difference_":
                        q_value = (home_model_value - home_model_value_pre) - (
                                away_model_value - away_model_value_pre)
                    elif self.DIFFERENCE_TYPE == "front_difference_":
                        q_value = (home_model_value_nex - home_model_value) - (
                                away_model_value_nex - away_model_value)
                    elif self.DIFFERENCE_TYPE == "skip_difference_":
                        q_value = (home_model_value_nex - home_model_value_pre) - (
                                away_model_value_nex - away_model_value_pre)
                    else:
                        raise ValueError('unknown difference type')
                    if player_value is None:
                        self.player_id_dict_all_by_match.update({playerId: {"value": q_value}})
                    else:
                        player_value_number = player_value.get("value") + q_value
                        self.player_id_dict_all_by_match.update(
                            {playerId: {"value": player_value_number}})
                    # "state value": model_state_value[0] - model_state_value[1]}})
                else:

                    if self.DIFFERENCE_TYPE == "back_difference_":
                        q_value = (away_model_value - away_model_value_pre) - (
                                home_model_value - home_model_value_pre)
                    elif self.DIFFERENCE_TYPE == "front_difference_":
                        q_value = (away_model_value_nex - away_model_value) - (
                                home_model_value_nex - home_model_value)
                    elif self.DIFFERENCE_TYPE == "skip_difference_":
                        q_value = (away_model_value_nex - away_model_value_pre) - (
                                home_model_value_nex - home_model_value_pre
                        )
                    else:
                        raise ValueError('unknown difference type')

                    if player_value is None:
                        self.player_id_dict_all_by_match.update({playerId: {"value": q_value}})
                    else:
                        player_value_number = player_value.get("value") + q_value
                        self.player_id_dict_all_by_match.update(
                            {playerId: {"value": player_value_number}})

