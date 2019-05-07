import os
import scipy.io as sio
import json
import unicodedata
import td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools as tools


class PlayerImpact:
    def __init__(self, data_name, model_data_store_dir, game_data_dir, difference_type='back_difference_'):
        self.player_id_dict_all = {}
        self.player_name_dict_all = {}
        # self.PLAYER_ID_DICT_ALL = {}
        self.difference_type = difference_type
        self.data_name = data_name
        self.game_data_dir = game_data_dir
        self.model_data_store_dir = model_data_store_dir

    def save_player_impact(self):
        assert len(self.player_name_dict_all.keys()) > 0
        with open('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/'
                  'td_three_prediction_two_tower_lstm_v_correct_dir/'
                  'compute_impact/player_impact/soccer_player_GIM.json', 'w') as f:
            json.dump(self.player_name_dict_all, f)

    def transfer2player_name_dict(self, player_id_name_pair_dir):
        with open(player_id_name_pair_dir, 'r') as f:
            player_id_name_pair = json.load(f)

        ids = player_id_name_pair.keys()
        # player_name_GIM_dict = {}
        for id in ids:
            GIM = self.player_id_dict_all.get(id)
            name = player_id_name_pair.get(id)
            name_str = name.get('first_name') + ' ' + name.get('last_name')
            self.player_name_dict_all.update({name_str: {'id': id, 'GIM': GIM}})

    def aggregate_match_diff_values(self, dir_game):
        """compute impact"""
        for file_name in os.listdir(self.model_data_store_dir + "/" + dir_game):
            if file_name == self.data_name:
                model_data_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                with open(model_data_name) as f:
                    model_data = json.load(f)
                    # model_data = (sio.loadmat(model_data_name))[self.data_name]
                    # elif file_name.startswith("playerId"):
                    #     playerIds_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                    #     playerIds = (sio.loadmat(playerIds_name))["playerId"][0]
                    # elif file_name.startswith("teamId"):
                    #     teamIds_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                    # teamIds = (sio.loadmat(teamIds_name))["teamId"][0]
            elif file_name.startswith("home_away"):
                home_identifier_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                home_identifier = (sio.loadmat(home_identifier_name))["home_away"][0]
            else:
                continue
        # TODO: fix the names of features
        actions = tools.read_feature_within_events(data_path=self.game_data_dir, directory=dir_game + '.json',
                                                   feature_name='action')
        # print actions
        playerIds = tools.read_feature_within_events(data_path=self.game_data_dir, directory=dir_game + '.json',
                                                     feature_name='playerId')
        # teamIds = tools.read_feature_within_events(data_path=self.game_data_dir, directory=dir_game+'.json',
        #                                           feature_name='teamIds')
        # print len(playerIds)
        # print len(actions)
        for player_Index in range(0, len(playerIds)):
            playerId = playerIds[player_Index]
            # teamId = teamIds[player_Index]
            # if int(teamId_target) == int(teamId):
            # print model_data
            model_value = model_data[str(player_Index)]
            if player_Index - 1 >= 0:  # define model pre
                if actions[player_Index - 1] == "goal":
                    model_value_pre = model_data[str(player_Index)]  # the goal cancel out here, just as we cut the game
                else:
                    model_value_pre = model_data[str(player_Index - 1)]
            else:
                model_value_pre = model_data[str(player_Index)]
            if player_Index + 1 < len(playerIds):  # define model next
                if actions[player_Index + 1] == "goal":
                    model_value_nex = model_data[str(player_Index)]
                else:
                    model_value_nex = model_data[str(player_Index + 1)]
            else:
                model_value_nex = model_data[str(player_Index)]

            ishome = home_identifier[player_Index]
            player_value = self.player_id_dict_all.get(playerId)

            home_model_value = model_value['home']
            away_model_value = model_value['away']
            # end_model_value = abs(model_value[2])
            home_model_value_pre = model_value_pre['home']
            away_model_value_pre = model_value_pre['away']
            # end_model_value_pre = abs(model_value_pre[2])
            home_model_value_nex = model_value_nex['home']
            away_model_value_nex = model_value_nex['away']
            # end_model_value_nex = abs(model_value_nex[2])

            if ishome:
                if self.difference_type == "back_difference_":
                    q_value = (home_model_value - home_model_value_pre)
                    # - (away_model_value - away_model_value_pre)
                elif self.difference_type == "front_difference_":
                    q_value = (home_model_value_nex - home_model_value)
                    # - (away_model_value_nex - away_model_value)
                elif self.difference_type == "skip_difference_":
                    q_value = (home_model_value_nex - home_model_value_pre)
                    # - (away_model_value_nex - away_model_value_pre)
                else:
                    raise ValueError('unknown difference type')
                if player_value is None:
                    self.player_id_dict_all.update({playerId: {"value": q_value}})
                else:
                    player_value_number = player_value.get("value") + q_value
                    self.player_id_dict_all.update(
                        {playerId: {"value": player_value_number}})
                    # "state value": model_state_value[0] - model_state_value[1]}})
            else:

                if self.difference_type == "back_difference_":
                    q_value = (away_model_value - away_model_value_pre)
                    # - (home_model_value - home_model_value_pre)
                elif self.difference_type == "front_difference_":
                    q_value = (away_model_value_nex - away_model_value)
                    # - (home_model_value_nex - home_model_value)
                elif self.difference_type == "skip_difference_":
                    q_value = (away_model_value_nex - away_model_value_pre)
                    # - (home_model_value_nex - home_model_value_pre)
                else:
                    raise ValueError('unknown difference type')

                if player_value is None:
                    self.player_id_dict_all.update({playerId: {"value": q_value}})
                else:
                    player_value_number = player_value.get("value") + q_value
                    self.player_id_dict_all.update(
                        {playerId: {"value": player_value_number}})
