import os
import scipy.io as sio
import unicodedata
import td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools as tools


class PlayerImpact:

    def __init__(self):
        self.player_id_dict_all_by_match = {}
        self.PLAYER_ID_DICT_ALL = {}
        self.difference_type = None
        self.data_name = None
        self.model_data_store_dir = None
        pass

    def aggregate_match_diff_values(self, dir_game, teamId_target):
        for file_name in os.listdir(self.model_data_store_dir + "/" + dir_game):
            if file_name == self.data_name + ".mat":
                model_data_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                model_data = (sio.loadmat(model_data_name))[self.data_name]
                # elif file_name.startswith("playerId"):
                #     playerIds_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                #     playerIds = (sio.loadmat(playerIds_name))["playerId"][0]
                # elif file_name.startswith("teamId"):
                #     teamIds_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                # teamIds = (sio.loadmat(teamIds_name))["teamId"][0]
            elif file_name.startswith("home_away"):
                home_identifier_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                home_identifier = (sio.loadmat(home_identifier_name))["home_away_"][0]
            else:
                continue
        # TODO: fix the names of features
        actions = tools.read_feature_within_events(data_path=self.model_data_store_dir, directory=dir_game,
                                                   feature_name='action')
        playerIds = tools.read_feature_within_events(data_path=self.model_data_store_dir, directory=dir_game,
                                                     feature_name='playerid')
        teamIds = tools.read_feature_within_events(data_path=self.model_data_store_dir, directory=dir_game,
                                                   feature_name='teamIds')
        for player_Index in range(0, len(playerIds)):
            playerId = playerIds[player_Index]
            teamId = teamIds[player_Index]
            if int(teamId_target) == int(teamId):
                model_value = model_data[player_Index]
                if player_Index - 1 >= 0:  # define model pre
                    if actions[player_Index - 1] == "goal":
                        model_value_pre = model_data[player_Index]  # the goal cancel out here, just as we cut the game
                    else:
                        model_value_pre = model_data[player_Index - 1]
                else:
                    model_value_pre = model_data[player_Index]
                if player_Index + 1 <= len(playerIds):  # define model next
                    if actions[player_Index + 1] == "goal":
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
                    if self.difference_type == "back_difference_":
                        q_value = (home_model_value - home_model_value_pre) - (
                                away_model_value - away_model_value_pre)
                    elif self.difference_type == "front_difference_":
                        q_value = (home_model_value_nex - home_model_value) - (
                                away_model_value_nex - away_model_value)
                    elif self.difference_type == "skip_difference_":
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

                    if self.difference_type == "back_difference_":
                        q_value = (away_model_value - away_model_value_pre) - (
                                home_model_value - home_model_value_pre)
                    elif self.difference_type == "front_difference_":
                        q_value = (away_model_value_nex - away_model_value) - (
                                home_model_value_nex - home_model_value)
                    elif self.difference_type == "skip_difference_":
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
